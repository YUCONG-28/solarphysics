$toolRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent (Split-Path -Parent $toolRoot)
$modulePath = Join-Path $toolRoot "LiteratureCatalog.psm1"
$configPath = Join-Path $toolRoot "config\paper_search_config.json"
$catalogJsonPath = Join-Path $repoRoot "Paper\catalog\papers.json"
$catalogMarkdownPath = Join-Path $repoRoot "Paper\catalog\papers.md"

Import-Module $modulePath -Force -DisableNameChecking

Describe "Literature catalog paths and schema" {
    It "resolves the repository catalog from the tool directory" {
        $paths = Resolve-LiteratureCatalogPaths -ToolRoot $toolRoot
        $paths.ProjectRoot | Should Be $repoRoot
        $paths.CatalogJsonPath | Should Be $catalogJsonPath
        $paths.CatalogMarkdownPath | Should Be $catalogMarkdownPath
    }

    It "keeps the repository catalog on the canonical 35-field schema without duplicates" {
        $parsed = Get-Content -Raw -Encoding UTF8 -LiteralPath $catalogJsonPath | ConvertFrom-Json
        $papers = @($parsed)
        $papers.Count | Should BeGreaterThan 0
        (Get-LiteratureCatalogFieldNames).Count | Should Be 35
        Assert-LiteratureCatalog -Papers $papers | Should Be $true
    }

    It "rejects duplicate title, DOI, and arXiv identities" {
        $parsed = Get-Content -Raw -Encoding UTF8 -LiteralPath $catalogJsonPath | ConvertFrom-Json
        $first = $parsed[0]
        { Assert-LiteratureCatalog -Papers @($first, $first) } | Should Throw
    }

    It "rejects non-string scalar fields and non-array collection fields" {
        $parsed = Get-Content -Raw -Encoding UTF8 -LiteralPath $catalogJsonPath | ConvertFrom-Json
        $badScalar = ($parsed[0] | ConvertTo-Json -Depth 8 | ConvertFrom-Json)
        $badScalar.year = 2026
        { Assert-LiteratureCatalog -Papers @($badScalar) } | Should Throw

        $badArray = ($parsed[0] | ConvertTo-Json -Depth 8 | ConvertFrom-Json)
        $badArray.authors = "not-an-array"
        { Assert-LiteratureCatalog -Papers @($badArray) } | Should Throw

        $badArrayEntry = ($parsed[0] | ConvertTo-Json -Depth 8 | ConvertFrom-Json)
        $badArrayEntry.topic_tags = @(42)
        { Assert-LiteratureCatalog -Papers @($badArrayEntry) } | Should Throw
    }

    It "rejects invalid relevance, priority, and catalog dates" {
        $parsed = Get-Content -Raw -Encoding UTF8 -LiteralPath $catalogJsonPath | ConvertFrom-Json
        $badRelevance = ($parsed[0] | ConvertTo-Json -Depth 8 | ConvertFrom-Json)
        $badRelevance.relevance_level = "Z"
        { Assert-LiteratureCatalog -Papers @($badRelevance) } | Should Throw

        $badPriority = ($parsed[0] | ConvertTo-Json -Depth 8 | ConvertFrom-Json)
        $badPriority.recommended_priority = "urgent"
        { Assert-LiteratureCatalog -Papers @($badPriority) } | Should Throw

        $badDate = ($parsed[0] | ConvertTo-Json -Depth 8 | ConvertFrom-Json)
        $badDate.last_checked = "2026/07/15"
        { Assert-LiteratureCatalog -Papers @($badDate) } | Should Throw
    }
}

Describe "Scoring and merge behavior" {
    It "marks Gaussian centroid imaging work as high relevance" {
        $config = Get-PaperSearchConfig -ConfigPath $configPath
        $rawPaper = [pscustomobject]@{
            title = "On the Source Position of a Solar Type III Radio Burst"
            authors = @("A. Author")
            year = "2026"
            journal = "arXiv preprint"
            summary = "LOFAR radio source centroid motion measured with 2D Gaussian fitting and FWHM."
            topic_tags = @("type III radio burst")
            method_tags = @("source trajectory")
        }

        $record = ConvertTo-ScoredPaperRecord -Paper $rawPaper -Config $config -DefaultDate "2026-07-15"
        $record.relevance_level | Should Be "A"
        $record.uses_gaussian_fitting | Should Be $true
        $record.uses_radio_centroid | Should Be $true
        $record.uses_frequency_drift_rate | Should Be $true
    }

    It "deduplicates normalized titles and preserves combined tags" {
        $first = [pscustomobject]@{ title = "Type-III solar radio bursts"; topic_tags = @("type III"); method_tags = @(); authors = @(); relevance_level = "B" }
        $second = [pscustomobject]@{ title = "Type III solar radio bursts"; topic_tags = @("radio spikes"); method_tags = @(); authors = @(); relevance_level = "A" }
        $merged = @(Merge-PaperRecords -PaperRecords @($first, $second))

        $merged.Count | Should Be 1
        $merged[0].relevance_level | Should Be "A"
        ($merged[0].topic_tags -join ";") | Should Match "type III"
        ($merged[0].topic_tags -join ";") | Should Match "radio spikes"
    }

    It "deduplicates by DOI and upgrades preprint metadata to a formal venue" {
        $preprint = [pscustomobject]@{
            title = "A preprint title"; journal = "arXiv preprint"; doi = ""; arxiv_id = "2601.00001"
            topic_tags = @(); method_tags = @(); authors = @(); relevance_level = "B"; last_checked = "2026-07-01"
        }
        $published = [pscustomobject]@{
            title = "A preprint title"; journal = "Astronomy & Astrophysics"; doi = "10.1000/example"; arxiv_id = "2601.00001v2"
            topic_tags = @(); method_tags = @(); authors = @(); relevance_level = "B"; last_checked = "2026-07-15"
        }
        $merged = @(Merge-PaperRecords -PaperRecords @($preprint, $published))

        $merged.Count | Should Be 1
        $merged[0].doi | Should Be "10.1000/example"
        $merged[0].journal | Should Be "Astronomy & Astrophysics"
        $merged[0].last_checked | Should Be "2026-07-15"
    }

    It "does not treat a placeholder Crossref venue plus DOI as a formal publication" {
        $preprint = [pscustomobject]@{
            title = "A preprint title"; journal = "arXiv preprint"; doi = ""; arxiv_id = "2601.00001"
            topic_tags = @(); method_tags = @(); authors = @(); relevance_level = "B"; last_checked = "2026-07-01"
        }
        $placeholder = [pscustomobject]@{
            title = "A preprint title"; journal = "Crossref result"; doi = "10.1000/example"; arxiv_id = ""
            topic_tags = @(); method_tags = @(); authors = @(); relevance_level = "B"; last_checked = "2026-07-15"
        }
        $merged = @(Merge-PaperRecords -PaperRecords @($preprint, $placeholder))

        $merged.Count | Should Be 1
        $merged[0].journal | Should Be "arXiv preprint"
        $merged[0].doi | Should Be ""
        $merged[0].last_checked | Should Be "2026-07-15"
    }
}

Describe "Live-search date window and source status" {
    It "accepts only dates inside the inclusive lookback window" {
        $module = Get-Module LiteratureCatalog
        $onBoundary = & $module { Test-CandidateDateWithinWindow -CandidateDate "2026-05-31" -AsOfDate "2026-07-15" -LookbackDays 45 }
        $tooOld = & $module { Test-CandidateDateWithinWindow -CandidateDate "2026-05-30" -AsOfDate "2026-07-15" -LookbackDays 45 }
        $future = & $module { Test-CandidateDateWithinWindow -CandidateDate "2026-07-16" -AsOfDate "2026-07-15" -LookbackDays 45 }

        $onBoundary | Should Be $true
        $tooOld | Should Be $false
        $future | Should Be $false
    }

    It "allows one source to succeed and reports the other source failure" {
        $tempConfig = Join-Path $TestDrive "partial-source-config.json"
        $config = Get-PaperSearchConfig -ConfigPath $configPath
        foreach ($group in @($config.keyword_groups)) { $group.queries = @("status probe") }
        $config | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 -LiteralPath $tempConfig

        $tempJson = Join-Path $TestDrive "partial-source-papers.json"
        $tempMarkdown = Join-Path $TestDrive "partial-source-papers.md"
        Copy-Item -LiteralPath $catalogJsonPath -Destination $tempJson
        Copy-Item -LiteralPath $catalogMarkdownPath -Destination $tempMarkdown

        Mock Invoke-WebRequest {
            [pscustomobject]@{ Content = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>' }
        } -ModuleName LiteratureCatalog
        Mock Invoke-RestMethod { throw "Crossref unavailable" } -ModuleName LiteratureCatalog

        $result = Invoke-LiteratureCatalogUpdate -ConfigPath $tempConfig -CatalogJsonPath $tempJson -CatalogMarkdownPath $tempMarkdown -AsOfDate "2026-07-15"

        $result.LiveSearchUsed | Should Be $true
        $result.LiveSearchSucceeded | Should Be $true
        $result.SourceStatus.Arxiv.State | Should Be "Succeeded"
        $result.SourceStatus.Arxiv.SuccessfulQueries | Should Be 1
        $result.SourceStatus.Crossref.State | Should Be "Failed"
        $result.SourceStatus.Crossref.FailedQueries | Should Be 1
    }

    It "fails a live run when both sources complete zero successful queries" {
        $tempConfig = Join-Path $TestDrive "failed-source-config.json"
        $config = Get-PaperSearchConfig -ConfigPath $configPath
        foreach ($group in @($config.keyword_groups)) { $group.queries = @("status probe") }
        $config | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 -LiteralPath $tempConfig

        $tempJson = Join-Path $TestDrive "failed-source-papers.json"
        $tempMarkdown = Join-Path $TestDrive "failed-source-papers.md"
        Copy-Item -LiteralPath $catalogJsonPath -Destination $tempJson
        Copy-Item -LiteralPath $catalogMarkdownPath -Destination $tempMarkdown

        Mock Invoke-WebRequest { throw "arXiv unavailable" } -ModuleName LiteratureCatalog
        Mock Invoke-RestMethod { throw "Crossref unavailable" } -ModuleName LiteratureCatalog

        {
            Invoke-LiteratureCatalogUpdate -ConfigPath $tempConfig -CatalogJsonPath $tempJson -CatalogMarkdownPath $tempMarkdown -AsOfDate "2026-07-15"
        } | Should Throw
    }
}

Describe "Deterministic render and offline check" {
    It "reproduces the stored six-column Markdown view" {
        $parsed = Get-Content -Raw -Encoding UTF8 -LiteralPath $catalogJsonPath | ConvertFrom-Json
        $expected = New-LiteratureCatalogMarkdown -Papers @($parsed)
        $stored = Get-Content -Raw -Encoding UTF8 -LiteralPath $catalogMarkdownPath
        (Test-TextContentEqual -Expected $expected -Actual $stored) | Should Be $true
    }

    It "checks a temporary catalog without changing either file" {
        $tempCatalog = Join-Path $TestDrive "catalog"
        New-Item -ItemType Directory -Path $tempCatalog -Force | Out-Null
        $tempJson = Join-Path $tempCatalog "papers.json"
        $tempMarkdown = Join-Path $tempCatalog "papers.md"
        Copy-Item -LiteralPath $catalogJsonPath -Destination $tempJson
        Copy-Item -LiteralPath $catalogMarkdownPath -Destination $tempMarkdown
        $beforeJson = (Get-FileHash -Algorithm SHA256 -LiteralPath $tempJson).Hash
        $beforeMarkdown = (Get-FileHash -Algorithm SHA256 -LiteralPath $tempMarkdown).Hash

        $result = Invoke-LiteratureCatalogUpdate -ConfigPath $configPath -CatalogJsonPath $tempJson -CatalogMarkdownPath $tempMarkdown -AsOfDate "2026-07-15" -Check

        $result.Mode | Should Be "Check"
        $result.LiveSearchUsed | Should Be $false
        (Get-FileHash -Algorithm SHA256 -LiteralPath $tempJson).Hash | Should Be $beforeJson
        (Get-FileHash -Algorithm SHA256 -LiteralPath $tempMarkdown).Hash | Should Be $beforeMarkdown
    }

    It "writes only papers.json and papers.md in offline update mode" {
        $tempCatalog = Join-Path $TestDrive "update-catalog"
        New-Item -ItemType Directory -Path $tempCatalog -Force | Out-Null
        $tempJson = Join-Path $tempCatalog "papers.json"
        $tempMarkdown = Join-Path $tempCatalog "papers.md"
        Copy-Item -LiteralPath $catalogJsonPath -Destination $tempJson

        $result = Invoke-LiteratureCatalogUpdate -ConfigPath $configPath -CatalogJsonPath $tempJson -CatalogMarkdownPath $tempMarkdown -AsOfDate "2026-07-15" -SkipLiveSearch
        $files = @(Get-ChildItem -File -LiteralPath $tempCatalog)

        $result.LiveSearchUsed | Should Be $false
        $files.Count | Should Be 2
        @($files.Name | Sort-Object) -join "," | Should Be "papers.json,papers.md"
    }
}

Describe "Two-file Git publish allowlist" {
    It "stages exactly the JSON catalog and Markdown view" {
        $plan = Get-GitLiteraturePublishPlan -AsOfDate "2026-07-15" -ChangedPaths @("Paper/catalog/papers.json", "Paper/catalog/papers.md")
        $plan.HasChanges | Should Be $true
        $plan.Pathspecs.Count | Should Be 2
        ($plan.Commands -join "`n") | Should Match 'git add -- "Paper/catalog/papers.json" "Paper/catalog/papers.md"'
        ($plan.Commands -join "`n") | Should Not Match "git add -A"
    }

    It "refuses all pre-existing staged paths" {
        { Get-GitLiteraturePublishPlan -AsOfDate "2026-07-15" -ChangedPaths @("Paper/catalog/papers.json") -StagedPaths @("Paper/catalog/papers.json") } | Should Throw
    }

    It "refuses unstaged changes outside the two catalog files" {
        { Get-GitLiteraturePublishPlan -AsOfDate "2026-07-15" -ChangedPaths @("Paper/catalog/papers.json", "README.md") } | Should Throw
    }
}
