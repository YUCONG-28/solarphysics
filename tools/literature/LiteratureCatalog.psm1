Set-StrictMode -Version Latest

$script:CatalogFieldNames = @(
    "citekey",
    "previous_citekeys",
    "title",
    "authors",
    "year",
    "journal",
    "doi",
    "arxiv_id",
    "ads_url",
    "pdf_url",
    "local_pdf_path",
    "local_pdf_sha256",
    "summary",
    "summary_cn",
    "topic_tags",
    "method_tags",
    "relevance_level",
    "recommended_priority",
    "key_methods",
    "why_relevant_to_current_project",
    "date_added",
    "last_checked",
    "direction",
    "instrument",
    "radio_frequency_range",
    "target_event_type",
    "gaussian_model",
    "centroid_definition",
    "source_size_definition",
    "beam_handling",
    "background_handling",
    "uncertainty_method",
    "quality_control",
    "applicability_to_DART_DRAT",
    "notes_for_current_code"
)

function Get-LiteratureCatalogFieldNames {
    [CmdletBinding()]
    param()

    return @($script:CatalogFieldNames)
}

function Resolve-LiteratureCatalogPaths {
    [CmdletBinding()]
    param(
        [string]$ToolRoot = $PSScriptRoot
    )

    $resolvedToolRoot = [System.IO.Path]::GetFullPath($ToolRoot)
    $toolsRoot = Split-Path -Parent $resolvedToolRoot
    $projectRoot = Split-Path -Parent $toolsRoot

    return [pscustomobject]@{
        ProjectRoot = $projectRoot
        ToolRoot = $resolvedToolRoot
        ConfigPath = Join-Path $resolvedToolRoot "config\paper_search_config.json"
        CatalogJsonPath = Join-Path $projectRoot "Paper\catalog\papers.json"
        CatalogMarkdownPath = Join-Path $projectRoot "Paper\catalog\papers.md"
    }
}

function Get-PaperSearchConfig {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ConfigPath
    )

    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        throw "Literature search config not found: $ConfigPath"
    }

    return Get-Content -Raw -Encoding UTF8 -LiteralPath $ConfigPath | ConvertFrom-Json
}

function Get-PropertyValue {
    [CmdletBinding()]
    param(
        [psobject]$Object,
        [string]$Name,
        $Default = $null
    )

    if ($null -eq $Object) {
        return $Default
    }

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $Default
    }

    return $property.Value
}

function Set-PropertyValue {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$Object,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        $Value
    )

    if ($null -eq $Object.PSObject.Properties[$Name]) {
        $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    }
    else {
        $Object.$Name = $Value
    }
}

function Merge-StringArray {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline = $true)]
        [object[]]$Values
    )

    $items = New-Object System.Collections.Generic.List[string]
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)

    foreach ($value in @($Values)) {
        if ($null -eq $value) {
            continue
        }

        $entries = if ($value -is [string]) {
            @($value -split ';')
        }
        else {
            @($value)
        }

        foreach ($entry in $entries) {
            if ($null -eq $entry) {
                continue
            }

            $trimmed = ([string]$entry).Trim()
            if ($trimmed -and $seen.Add($trimmed)) {
                [void]$items.Add($trimmed)
            }
        }
    }

    return @($items.ToArray())
}

function Get-NormalizedTitle {
    [CmdletBinding()]
    param([string]$Title)

    if (-not $Title) {
        return ""
    }

    $lower = $Title.ToLowerInvariant()
    return ([regex]::Replace($lower, '[^a-z0-9]+', ' ')).Trim()
}

function Get-NormalizedDoi {
    [CmdletBinding()]
    param([string]$Doi)

    if (-not $Doi) {
        return ""
    }

    $normalized = $Doi.Trim().ToLowerInvariant()
    $normalized = $normalized -replace '^https?://(dx\.)?doi\.org/', ''
    $normalized = $normalized -replace '^doi:\s*', ''
    return $normalized.Trim()
}

function Get-NormalizedArxivId {
    [CmdletBinding()]
    param([string]$ArxivId)

    if (-not $ArxivId) {
        return ""
    }

    $normalized = $ArxivId.Trim().ToLowerInvariant()
    $normalized = $normalized -replace '^https?://arxiv\.org/(abs|pdf)/', ''
    $normalized = $normalized -replace '\.pdf$', ''
    $normalized = $normalized -replace '^arxiv:\s*', ''
    $normalized = $normalized -replace 'v\d+$', ''
    return $normalized.Trim('/')
}

function Get-PaperIdentityKeys {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$Paper
    )

    $keys = New-Object System.Collections.Generic.List[string]
    $title = Get-NormalizedTitle -Title ([string](Get-PropertyValue -Object $Paper -Name "title" -Default ""))
    $citekey = ([string](Get-PropertyValue -Object $Paper -Name "citekey" -Default "")).Trim().ToLowerInvariant()
    $doi = Get-NormalizedDoi -Doi ([string](Get-PropertyValue -Object $Paper -Name "doi" -Default ""))
    $arxivId = Get-NormalizedArxivId -ArxivId ([string](Get-PropertyValue -Object $Paper -Name "arxiv_id" -Default ""))

    if ($citekey) { [void]$keys.Add("citekey:$citekey") }
    if ($title) { [void]$keys.Add("title:$title") }
    if ($doi) { [void]$keys.Add("doi:$doi") }
    if ($arxivId) { [void]$keys.Add("arxiv:$arxivId") }

    return @($keys.ToArray())
}

function Test-TextMatch {
    [CmdletBinding()]
    param(
        [string]$Text,
        [string]$Pattern
    )

    if (-not $Text -or -not $Pattern) {
        return $false
    }

    return $Text.ToLowerInvariant().Contains($Pattern.ToLowerInvariant())
}

function Get-RelevanceRank {
    [CmdletBinding()]
    param([string]$RelevanceLevel)

    switch ($RelevanceLevel) {
        "A" { return 4 }
        "B" { return 3 }
        "C" { return 2 }
        "D" { return 1 }
        default { return 0 }
    }
}

function Get-PaperTextBlob {
    [CmdletBinding()]
    param([psobject]$Paper)

    $chunks = @(
        (Get-PropertyValue -Object $Paper -Name "title" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "summary" -Default ""),
        ((Merge-StringArray -Values @(
            (Get-PropertyValue -Object $Paper -Name "topic_tags" -Default @()),
            (Get-PropertyValue -Object $Paper -Name "method_tags" -Default @())
        )) -join " "),
        (Get-PropertyValue -Object $Paper -Name "journal" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "instrument" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "target_event_type" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "gaussian_model" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "centroid_definition" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "source_size_definition" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "beam_handling" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "background_handling" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "uncertainty_method" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "quality_control" -Default ""),
        (Get-PropertyValue -Object $Paper -Name "applicability_to_DART_DRAT" -Default "")
    )

    return (($chunks | Where-Object { $_ }) -join " ").ToLowerInvariant()
}

function Get-RecommendedPriority {
    [CmdletBinding()]
    param(
        [string]$RelevanceLevel,
        $Config
    )

    $configured = Get-PropertyValue -Object $Config.priority_map -Name $RelevanceLevel
    if ($configured) {
        return [string]$configured
    }

    switch ($RelevanceLevel) {
        "A" { return "high" }
        "B" { return "medium" }
        "C" { return "background" }
        default { return "hold" }
    }
}

function ConvertTo-ScoredPaperRecord {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$Paper,
        [Parameter(Mandatory = $true)]
        $Config,
        [string]$DefaultDate = (Get-Date -Format "yyyy-MM-dd")
    )

    $text = Get-PaperTextBlob -Paper $Paper
    $topicTags = @(Merge-StringArray -Values @((Get-PropertyValue -Object $Paper -Name "topic_tags" -Default @())))
    $methodTags = @(Merge-StringArray -Values @((Get-PropertyValue -Object $Paper -Name "method_tags" -Default @())))
    $score = 0
    $matchedDirections = New-Object System.Collections.Generic.List[string]

    foreach ($group in @($Config.keyword_groups)) {
        $groupMatched = $false
        foreach ($term in @($group.score_terms)) {
            if (Test-TextMatch -Text $text -Pattern $term.pattern) {
                $score += [int]$term.score
                $groupMatched = $true
            }
        }

        if ($groupMatched) {
            [void]$matchedDirections.Add([string]$group.direction)
        }
    }

    $flagResults = @{}
    foreach ($definition in @(
        @{ Name = "uses_gaussian_fitting"; Patterns = $Config.flags.uses_gaussian_fitting },
        @{ Name = "uses_radio_centroid"; Patterns = $Config.flags.uses_radio_centroid },
        @{ Name = "uses_source_size"; Patterns = $Config.flags.uses_source_size },
        @{ Name = "uses_beam_deconvolution"; Patterns = $Config.flags.uses_beam_deconvolution },
        @{ Name = "uses_aia"; Patterns = $Config.flags.uses_aia },
        @{ Name = "uses_hmi"; Patterns = $Config.flags.uses_hmi },
        @{ Name = "uses_newkirk_model"; Patterns = $Config.flags.uses_newkirk_model },
        @{ Name = "uses_frequency_drift_rate"; Patterns = $Config.flags.uses_frequency_drift_rate }
    )) {
        $matched = $false
        foreach ($pattern in @($definition.Patterns)) {
            if (Test-TextMatch -Text $text -Pattern $pattern) {
                $matched = $true
                break
            }
        }
        $flagResults[$definition.Name] = $matched
    }

    if (-not $flagResults.uses_frequency_drift_rate -and
        (Test-TextMatch -Text $text -Pattern "type iii") -and
        ((Test-TextMatch -Text $text -Pattern "duration") -or (Test-TextMatch -Text $text -Pattern "motion"))) {
        $flagResults.uses_frequency_drift_rate = $true
    }

    if ($flagResults.uses_gaussian_fitting -and $flagResults.uses_radio_centroid) { $score += 4 }
    if ($flagResults.uses_source_size) { $score += 2 }
    if ($flagResults.uses_beam_deconvolution) { $score += 2 }
    if ($flagResults.uses_aia -or $flagResults.uses_hmi) { $score += 2 }
    if ($flagResults.uses_frequency_drift_rate -or $flagResults.uses_newkirk_model) { $score += 2 }

    $relevanceLevel = [string](Get-PropertyValue -Object $Paper -Name "relevance_level" -Default "")
    if (-not $relevanceLevel) {
        if ($score -ge [int]$Config.relevance_thresholds.A) { $relevanceLevel = "A" }
        elseif ($score -ge [int]$Config.relevance_thresholds.B) { $relevanceLevel = "B" }
        elseif ($score -ge [int]$Config.relevance_thresholds.C) { $relevanceLevel = "C" }
        else { $relevanceLevel = "D" }
    }

    $priority = [string](Get-PropertyValue -Object $Paper -Name "recommended_priority" -Default "")
    if (-not $priority) {
        $priority = Get-RecommendedPriority -RelevanceLevel $relevanceLevel -Config $Config
    }

    $topicTags = @(Merge-StringArray -Values @($topicTags, $matchedDirections.ToArray()))
    $methodTags = @(Merge-StringArray -Values @(
        $methodTags,
        @(
            $(if ($flagResults.uses_gaussian_fitting) { "Gaussian fitting" }),
            $(if ($flagResults.uses_radio_centroid) { "radio source centroid" }),
            $(if ($flagResults.uses_source_size) { "source size / FWHM" }),
            $(if ($flagResults.uses_beam_deconvolution) { "beam handling" }),
            $(if ($flagResults.uses_frequency_drift_rate) { "frequency drift rate" })
        )
    ))

    return [pscustomobject]@{
        title = [string](Get-PropertyValue -Object $Paper -Name "title" -Default "")
        authors = @(Merge-StringArray -Values @((Get-PropertyValue -Object $Paper -Name "authors" -Default @())))
        year = [string](Get-PropertyValue -Object $Paper -Name "year" -Default "")
        journal = [string](Get-PropertyValue -Object $Paper -Name "journal" -Default "")
        doi = [string](Get-PropertyValue -Object $Paper -Name "doi" -Default "")
        arxiv_id = [string](Get-PropertyValue -Object $Paper -Name "arxiv_id" -Default "")
        ads_url = [string](Get-PropertyValue -Object $Paper -Name "ads_url" -Default "")
        pdf_url = [string](Get-PropertyValue -Object $Paper -Name "pdf_url" -Default "")
        local_pdf_path = [string](Get-PropertyValue -Object $Paper -Name "local_pdf_path" -Default "")
        summary = [string](Get-PropertyValue -Object $Paper -Name "summary" -Default "")
        summary_cn = [string](Get-PropertyValue -Object $Paper -Name "summary_cn" -Default "Pending verification")
        topic_tags = $topicTags
        method_tags = $methodTags
        relevance_level = $relevanceLevel
        recommended_priority = $priority
        key_methods = [string](Get-PropertyValue -Object $Paper -Name "key_methods" -Default "")
        why_relevant_to_current_project = [string](Get-PropertyValue -Object $Paper -Name "why_relevant_to_current_project" -Default "")
        date_added = [string](Get-PropertyValue -Object $Paper -Name "date_added" -Default $DefaultDate)
        last_checked = [string](Get-PropertyValue -Object $Paper -Name "last_checked" -Default $DefaultDate)
        direction = [string](Get-PropertyValue -Object $Paper -Name "direction" -Default (($matchedDirections | Select-Object -First 1)))
        instrument = [string](Get-PropertyValue -Object $Paper -Name "instrument" -Default "")
        radio_frequency_range = [string](Get-PropertyValue -Object $Paper -Name "radio_frequency_range" -Default "")
        target_event_type = [string](Get-PropertyValue -Object $Paper -Name "target_event_type" -Default "")
        gaussian_model = [string](Get-PropertyValue -Object $Paper -Name "gaussian_model" -Default "")
        centroid_definition = [string](Get-PropertyValue -Object $Paper -Name "centroid_definition" -Default "")
        source_size_definition = [string](Get-PropertyValue -Object $Paper -Name "source_size_definition" -Default "")
        beam_handling = [string](Get-PropertyValue -Object $Paper -Name "beam_handling" -Default "")
        background_handling = [string](Get-PropertyValue -Object $Paper -Name "background_handling" -Default "")
        uncertainty_method = [string](Get-PropertyValue -Object $Paper -Name "uncertainty_method" -Default "")
        quality_control = [string](Get-PropertyValue -Object $Paper -Name "quality_control" -Default "")
        applicability_to_DART_DRAT = [string](Get-PropertyValue -Object $Paper -Name "applicability_to_DART_DRAT" -Default "")
        notes_for_current_code = [string](Get-PropertyValue -Object $Paper -Name "notes_for_current_code" -Default "")
        uses_gaussian_fitting = [bool]$flagResults.uses_gaussian_fitting
        uses_radio_centroid = [bool]$flagResults.uses_radio_centroid
        uses_source_size = [bool]$flagResults.uses_source_size
        uses_beam_deconvolution = [bool]$flagResults.uses_beam_deconvolution
        uses_aia = [bool]$flagResults.uses_aia
        uses_hmi = [bool]$flagResults.uses_hmi
        uses_newkirk_model = [bool]$flagResults.uses_newkirk_model
        uses_frequency_drift_rate = [bool]$flagResults.uses_frequency_drift_rate
    }
}

function Get-YearFromCrossrefItem {
    [CmdletBinding()]
    param($Item)

    foreach ($propertyName in @("published-print", "published-online", "issued", "created")) {
        $property = $Item.PSObject.Properties[$propertyName]
        if ($null -eq $property) { continue }
        $dateParts = $property.Value."date-parts"
        if ($dateParts -and $dateParts[0] -and $dateParts[0][0]) {
            return [string]$dateParts[0][0]
        }
    }

    return ""
}

function Get-DateFromCrossrefItem {
    [CmdletBinding()]
    param($Item)

    foreach ($propertyName in @("published-print", "published-online", "issued", "created")) {
        $property = $Item.PSObject.Properties[$propertyName]
        if ($null -eq $property) { continue }

        $dateParts = $property.Value."date-parts"
        if (-not $dateParts -or -not $dateParts[0] -or $dateParts[0].Count -lt 3) {
            continue
        }

        try {
            $date = New-Object datetime (
                [int]$dateParts[0][0],
                [int]$dateParts[0][1],
                [int]$dateParts[0][2]
            )
            return $date.ToString("yyyy-MM-dd", [System.Globalization.CultureInfo]::InvariantCulture)
        }
        catch {
            continue
        }
    }

    return ""
}

function Test-CandidateDateWithinWindow {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$CandidateDate,
        [Parameter(Mandatory = $true)][string]$AsOfDate,
        [Parameter(Mandatory = $true)][int]$LookbackDays
    )

    if ($LookbackDays -lt 0) {
        throw "lookback_days must be zero or greater."
    }

    $candidate = [datetime]::MinValue
    $asOf = [datetime]::MinValue
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $styles = [System.Globalization.DateTimeStyles]::None
    if (-not [datetime]::TryParseExact($CandidateDate, "yyyy-MM-dd", $culture, $styles, [ref]$candidate)) {
        return $false
    }
    if (-not [datetime]::TryParseExact($AsOfDate, "yyyy-MM-dd", $culture, $styles, [ref]$asOf)) {
        throw "-AsOfDate must use yyyy-MM-dd format."
    }

    $windowStart = $asOf.AddDays(-$LookbackDays)
    return $candidate -ge $windowStart -and $candidate -le $asOf
}

function Invoke-ArxivSearch {
    [CmdletBinding()]
    param(
        [string[]]$Queries,
        [int]$MaxResults,
        [string]$AsOfDate,
        [int]$LookbackDays
    )

    $results = New-Object System.Collections.Generic.List[object]
    $headers = @{ "User-Agent" = "solarphysics-literature-catalog/1.0" }
    $successfulQueries = 0
    $failedQueries = 0
    $rejectedByDate = 0

    foreach ($query in @($Queries)) {
        try {
            $searchQuery = "all:$query"
            $encodedQuery = [System.Uri]::EscapeDataString($searchQuery)
            $uri = "https://export.arxiv.org/api/query?search_query=$encodedQuery&start=0&max_results=$MaxResults&sortBy=lastUpdatedDate&sortOrder=descending"
            $xmlContent = (Invoke-WebRequest -Uri $uri -Headers $headers -UseBasicParsing).Content
            $feed = [xml]$xmlContent
            $successfulQueries += 1

            $entries = @($feed.SelectNodes("/*[local-name()='feed']/*[local-name()='entry']"))
            foreach ($entry in $entries) {
                $sourceTimestamp = if ([string]$entry.updated) { [string]$entry.updated } else { [string]$entry.published }
                $sourceDateValue = [datetimeoffset]::MinValue
                if (-not [datetimeoffset]::TryParse(
                    $sourceTimestamp,
                    [System.Globalization.CultureInfo]::InvariantCulture,
                    [System.Globalization.DateTimeStyles]::AssumeUniversal,
                    [ref]$sourceDateValue
                )) {
                    $rejectedByDate += 1
                    continue
                }
                $sourceDate = $sourceDateValue.UtcDateTime.ToString("yyyy-MM-dd", [System.Globalization.CultureInfo]::InvariantCulture)
                if (-not (Test-CandidateDateWithinWindow -CandidateDate $sourceDate -AsOfDate $AsOfDate -LookbackDays $LookbackDays)) {
                    $rejectedByDate += 1
                    continue
                }

                $authors = @($entry.author | ForEach-Object { [string]$_.name })
                $pdfLink = ""
                foreach ($link in @($entry.link)) {
                    if ($link.title -eq "pdf") {
                        $pdfLink = [string]$link.href
                        break
                    }
                }

                $identifier = [string]$entry.id
                $arxivId = if ($identifier) { ($identifier.TrimEnd('/') -split '/')[-1] } else { "" }
                $published = [string]$entry.published
                $year = if ($published.Length -ge 4) { $published.Substring(0, 4) } else { "" }
                [void]$results.Add([pscustomobject]@{
                    title = ([string]$entry.title).Trim()
                    authors = $authors
                    year = $year
                    journal = "arXiv preprint"
                    doi = ""
                    arxiv_id = $arxivId
                    ads_url = ""
                    pdf_url = $pdfLink
                    local_pdf_path = ""
                    summary = ([string]$entry.summary).Trim()
                    summary_cn = "Pending verification"
                    topic_tags = @("live search", "arXiv")
                    method_tags = @()
                    date_added = $AsOfDate
                    last_checked = $AsOfDate
                    source_date = $sourceDate
                })
            }
        }
        catch {
            $failedQueries += 1
            Write-Warning "arXiv query failed for $query : $($_.Exception.Message)"
        }
    }

    $state = if ($successfulQueries -gt 0 -and $failedQueries -gt 0) {
        "PartialSuccess"
    }
    elseif ($successfulQueries -gt 0) {
        "Succeeded"
    }
    else {
        "Failed"
    }
    return [pscustomobject]@{
        Source = "arXiv"
        Candidates = @($results.ToArray())
        CandidateCount = $results.Count
        SuccessfulQueries = $successfulQueries
        FailedQueries = $failedQueries
        RejectedByDate = $rejectedByDate
        Succeeded = $successfulQueries -gt 0
        State = $state
    }
}

function Invoke-CrossrefSearch {
    [CmdletBinding()]
    param(
        [string[]]$Queries,
        [int]$Rows,
        [string]$AsOfDate,
        [int]$LookbackDays
    )

    $results = New-Object System.Collections.Generic.List[object]
    $headers = @{ "User-Agent" = "solarphysics-literature-catalog/1.0" }
    $successfulQueries = 0
    $failedQueries = 0
    $rejectedByDate = 0

    foreach ($query in @($Queries)) {
        try {
            $encodedQuery = [System.Uri]::EscapeDataString($query)
            $uri = "https://api.crossref.org/works?query.title=$encodedQuery&rows=$Rows&sort=published&order=desc"
            $response = Invoke-RestMethod -Uri $uri -Headers $headers
            $successfulQueries += 1

            foreach ($item in @($response.message.items)) {
                $title = if ($item.title) { [string]$item.title[0] } else { "" }
                if (-not $title) { continue }

                $sourceDate = Get-DateFromCrossrefItem -Item $item
                if (-not $sourceDate -or -not (Test-CandidateDateWithinWindow -CandidateDate $sourceDate -AsOfDate $AsOfDate -LookbackDays $LookbackDays)) {
                    $rejectedByDate += 1
                    continue
                }

                $authors = @()
                foreach ($author in @($item.author)) {
                    $authorName = (($author.given, $author.family) -join " ").Trim()
                    if ($authorName) { $authors += $authorName }
                }

                [void]$results.Add([pscustomobject]@{
                    title = $title
                    authors = $authors
                    year = Get-YearFromCrossrefItem -Item $item
                    journal = if ($item."container-title") { [string]$item."container-title"[0] } else { "Crossref result" }
                    doi = [string]$item.DOI
                    arxiv_id = ""
                    ads_url = ""
                    pdf_url = ""
                    local_pdf_path = ""
                    summary = ""
                    summary_cn = "Pending verification"
                    topic_tags = @("live search", "Crossref")
                    method_tags = @()
                    date_added = $AsOfDate
                    last_checked = $AsOfDate
                    source_date = $sourceDate
                })
            }
        }
        catch {
            $failedQueries += 1
            Write-Warning "Crossref query failed for $query : $($_.Exception.Message)"
        }
    }

    $state = if ($successfulQueries -gt 0 -and $failedQueries -gt 0) {
        "PartialSuccess"
    }
    elseif ($successfulQueries -gt 0) {
        "Succeeded"
    }
    else {
        "Failed"
    }
    return [pscustomobject]@{
        Source = "Crossref"
        Candidates = @($results.ToArray())
        CandidateCount = $results.Count
        SuccessfulQueries = $successfulQueries
        FailedQueries = $failedQueries
        RejectedByDate = $rejectedByDate
        Succeeded = $successfulQueries -gt 0
        State = $state
    }
}

function Get-LivePaperCandidates {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Config,
        [Parameter(Mandatory = $true)]
        [string]$AsOfDate
    )

    $queries = New-Object System.Collections.Generic.List[string]
    foreach ($group in @($Config.keyword_groups)) {
        foreach ($query in @($group.queries)) {
            if ($query) { [void]$queries.Add([string]$query) }
        }
    }

    $uniqueQueries = @($queries.ToArray() | Sort-Object -Unique)
    $lookbackDays = 0
    if (-not [int]::TryParse([string]$Config.search_settings.lookback_days, [ref]$lookbackDays) -or $lookbackDays -lt 0) {
        throw "search_settings.lookback_days must be a non-negative integer."
    }

    $arxivResult = Invoke-ArxivSearch `
        -Queries $uniqueQueries `
        -MaxResults ([int]$Config.search_settings.max_results_per_query) `
        -AsOfDate $AsOfDate `
        -LookbackDays $lookbackDays
    $crossrefResult = Invoke-CrossrefSearch `
        -Queries $uniqueQueries `
        -Rows ([int]$Config.search_settings.crossref_rows) `
        -AsOfDate $AsOfDate `
        -LookbackDays $lookbackDays

    if (($arxivResult.SuccessfulQueries + $crossrefResult.SuccessfulQueries) -eq 0) {
        throw "Live literature search failed: arXiv and Crossref completed zero successful queries."
    }

    return [pscustomobject]@{
        Candidates = @($arxivResult.Candidates) + @($crossrefResult.Candidates)
        SourceStatus = [pscustomobject]@{
            Arxiv = $arxivResult
            Crossref = $crossrefResult
        }
    }
}

function Test-FormalPublicationRecord {
    [CmdletBinding()]
    param([psobject]$Paper)

    $journal = ([string](Get-PropertyValue -Object $Paper -Name "journal" -Default "")).Trim()
    return $journal -and $journal -notmatch '(?i)arxiv|preprint|crossref result|unknown|pending|not available'
}

function Merge-TwoPaperRecords {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$Existing,
        [Parameter(Mandatory = $true)]
        [psobject]$Incoming
    )

    $existingFormal = Test-FormalPublicationRecord -Paper $Existing
    $incomingFormal = Test-FormalPublicationRecord -Paper $Incoming
    $existingRelevance = [string](Get-PropertyValue -Object $Existing -Name "relevance_level" -Default "")
    $incomingRelevance = [string](Get-PropertyValue -Object $Incoming -Name "relevance_level" -Default "")

    foreach ($name in @("previous_citekeys", "authors", "topic_tags", "method_tags")) {
        $combined = @(Merge-StringArray -Values @(
            (Get-PropertyValue -Object $Existing -Name $name -Default @()),
            (Get-PropertyValue -Object $Incoming -Name $name -Default @())
        ))
        Set-PropertyValue -Object $Existing -Name $name -Value $combined
    }

    foreach ($name in @(
        "uses_gaussian_fitting",
        "uses_radio_centroid",
        "uses_source_size",
        "uses_beam_deconvolution",
        "uses_aia",
        "uses_hmi",
        "uses_newkirk_model",
        "uses_frequency_drift_rate"
    )) {
        $value = [bool](Get-PropertyValue -Object $Existing -Name $name -Default $false) -or
            [bool](Get-PropertyValue -Object $Incoming -Name $name -Default $false)
        Set-PropertyValue -Object $Existing -Name $name -Value $value
    }

    if ((Get-RelevanceRank -RelevanceLevel $incomingRelevance) -gt (Get-RelevanceRank -RelevanceLevel $existingRelevance)) {
        Set-PropertyValue -Object $Existing -Name "relevance_level" -Value $incomingRelevance
        $incomingPriority = [string](Get-PropertyValue -Object $Incoming -Name "recommended_priority" -Default "")
        if ($incomingPriority) {
            Set-PropertyValue -Object $Existing -Name "recommended_priority" -Value $incomingPriority
        }
    }

    $existingDateAdded = [string](Get-PropertyValue -Object $Existing -Name "date_added" -Default "")
    $incomingDateAdded = [string](Get-PropertyValue -Object $Incoming -Name "date_added" -Default "")
    if ($incomingDateAdded -and (-not $existingDateAdded -or $incomingDateAdded -lt $existingDateAdded)) {
        Set-PropertyValue -Object $Existing -Name "date_added" -Value $incomingDateAdded
    }

    $existingLastChecked = [string](Get-PropertyValue -Object $Existing -Name "last_checked" -Default "")
    $incomingLastChecked = [string](Get-PropertyValue -Object $Incoming -Name "last_checked" -Default "")
    if ($incomingLastChecked -and (-not $existingLastChecked -or $incomingLastChecked -gt $existingLastChecked)) {
        Set-PropertyValue -Object $Existing -Name "last_checked" -Value $incomingLastChecked
    }

    if ($incomingFormal) {
        $incomingJournal = [string](Get-PropertyValue -Object $Incoming -Name "journal" -Default "")
        $incomingDoi = [string](Get-PropertyValue -Object $Incoming -Name "doi" -Default "")
        $existingDoi = [string](Get-PropertyValue -Object $Existing -Name "doi" -Default "")
        if (-not $existingFormal -and $incomingJournal) {
            Set-PropertyValue -Object $Existing -Name "journal" -Value $incomingJournal
        }
        if (-not $existingDoi -and $incomingDoi) {
            Set-PropertyValue -Object $Existing -Name "doi" -Value $incomingDoi
        }
    }

    foreach ($property in @($Incoming.PSObject.Properties)) {
        $name = [string]$property.Name
        if ($name -in @(
            "previous_citekeys", "authors", "topic_tags", "method_tags", "relevance_level", "recommended_priority",
            "date_added", "last_checked", "journal", "doi",
            "uses_gaussian_fitting", "uses_radio_centroid", "uses_source_size",
            "uses_beam_deconvolution", "uses_aia", "uses_hmi", "uses_newkirk_model",
            "uses_frequency_drift_rate"
        )) {
            continue
        }

        $currentValue = Get-PropertyValue -Object $Existing -Name $name -Default $null
        $incomingValue = $property.Value
        $currentIsEmpty = $null -eq $currentValue -or
            ($currentValue -is [string] -and [string]::IsNullOrWhiteSpace([string]$currentValue))
        $incomingHasValue = $null -ne $incomingValue -and
            (-not ($incomingValue -is [string]) -or -not [string]::IsNullOrWhiteSpace([string]$incomingValue))

        if ($currentIsEmpty -and $incomingHasValue) {
            Set-PropertyValue -Object $Existing -Name $name -Value $incomingValue
        }
    }

    return $Existing
}

function Merge-PaperRecords {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$PaperRecords
    )

    $merged = New-Object System.Collections.Generic.List[object]

    foreach ($record in @($PaperRecords)) {
        if ($null -eq $record) { continue }

        $recordKeys = @(Get-PaperIdentityKeys -Paper $record)
        $matchingIndexes = New-Object System.Collections.Generic.List[int]
        for ($index = 0; $index -lt $merged.Count; $index++) {
            $existingKeys = @(Get-PaperIdentityKeys -Paper $merged[$index])
            $hasMatch = @($recordKeys | Where-Object { $existingKeys -contains $_ }).Count -gt 0
            if ($hasMatch) { [void]$matchingIndexes.Add($index) }
        }

        if ($matchingIndexes.Count -eq 0) {
            [void]$merged.Add($record)
            continue
        }

        $primaryIndex = $matchingIndexes[0]
        $primary = Merge-TwoPaperRecords -Existing $merged[$primaryIndex] -Incoming $record
        for ($matchOffset = $matchingIndexes.Count - 1; $matchOffset -ge 1; $matchOffset--) {
            $secondaryIndex = $matchingIndexes[$matchOffset]
            $primary = Merge-TwoPaperRecords -Existing $primary -Incoming $merged[$secondaryIndex]
            $merged.RemoveAt($secondaryIndex)
        }
        $merged[$primaryIndex] = $primary
    }

    return @($merged.ToArray())
}

function ConvertTo-CatalogPaper {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$Paper,
        [string]$DefaultDate = (Get-Date -Format "yyyy-MM-dd")
    )

    return [pscustomobject][ordered]@{
        citekey = [string](Get-PropertyValue -Object $Paper -Name "citekey" -Default "")
        previous_citekeys = @(Merge-StringArray -Values @((Get-PropertyValue -Object $Paper -Name "previous_citekeys" -Default @())))
        title = [string](Get-PropertyValue -Object $Paper -Name "title" -Default "")
        authors = @(Merge-StringArray -Values @((Get-PropertyValue -Object $Paper -Name "authors" -Default @())))
        year = [string](Get-PropertyValue -Object $Paper -Name "year" -Default "")
        journal = [string](Get-PropertyValue -Object $Paper -Name "journal" -Default "")
        doi = [string](Get-PropertyValue -Object $Paper -Name "doi" -Default "")
        arxiv_id = [string](Get-PropertyValue -Object $Paper -Name "arxiv_id" -Default "")
        ads_url = [string](Get-PropertyValue -Object $Paper -Name "ads_url" -Default "")
        pdf_url = [string](Get-PropertyValue -Object $Paper -Name "pdf_url" -Default "")
        local_pdf_path = [string](Get-PropertyValue -Object $Paper -Name "local_pdf_path" -Default "")
        local_pdf_sha256 = [string](Get-PropertyValue -Object $Paper -Name "local_pdf_sha256" -Default "")
        summary = [string](Get-PropertyValue -Object $Paper -Name "summary" -Default "")
        summary_cn = [string](Get-PropertyValue -Object $Paper -Name "summary_cn" -Default "Pending verification")
        topic_tags = @(Merge-StringArray -Values @((Get-PropertyValue -Object $Paper -Name "topic_tags" -Default @())))
        method_tags = @(Merge-StringArray -Values @((Get-PropertyValue -Object $Paper -Name "method_tags" -Default @())))
        relevance_level = [string](Get-PropertyValue -Object $Paper -Name "relevance_level" -Default "D")
        recommended_priority = [string](Get-PropertyValue -Object $Paper -Name "recommended_priority" -Default "hold")
        key_methods = [string](Get-PropertyValue -Object $Paper -Name "key_methods" -Default "")
        why_relevant_to_current_project = [string](Get-PropertyValue -Object $Paper -Name "why_relevant_to_current_project" -Default "")
        date_added = [string](Get-PropertyValue -Object $Paper -Name "date_added" -Default $DefaultDate)
        last_checked = [string](Get-PropertyValue -Object $Paper -Name "last_checked" -Default $DefaultDate)
        direction = [string](Get-PropertyValue -Object $Paper -Name "direction" -Default "")
        instrument = [string](Get-PropertyValue -Object $Paper -Name "instrument" -Default "")
        radio_frequency_range = [string](Get-PropertyValue -Object $Paper -Name "radio_frequency_range" -Default "")
        target_event_type = [string](Get-PropertyValue -Object $Paper -Name "target_event_type" -Default "")
        gaussian_model = [string](Get-PropertyValue -Object $Paper -Name "gaussian_model" -Default "")
        centroid_definition = [string](Get-PropertyValue -Object $Paper -Name "centroid_definition" -Default "")
        source_size_definition = [string](Get-PropertyValue -Object $Paper -Name "source_size_definition" -Default "")
        beam_handling = [string](Get-PropertyValue -Object $Paper -Name "beam_handling" -Default "")
        background_handling = [string](Get-PropertyValue -Object $Paper -Name "background_handling" -Default "")
        uncertainty_method = [string](Get-PropertyValue -Object $Paper -Name "uncertainty_method" -Default "")
        quality_control = [string](Get-PropertyValue -Object $Paper -Name "quality_control" -Default "")
        applicability_to_DART_DRAT = [string](Get-PropertyValue -Object $Paper -Name "applicability_to_DART_DRAT" -Default "")
        notes_for_current_code = [string](Get-PropertyValue -Object $Paper -Name "notes_for_current_code" -Default "")
    }
}

function Set-CatalogCitekeys {
    [CmdletBinding()]
    param([object[]]$Papers)

    $used = @{}
    foreach ($paper in @($Papers)) {
        foreach ($key in @([string]$paper.citekey) + @($paper.previous_citekeys)) {
            $trimmed = ([string]$key).Trim()
            if ($trimmed) { $used[$trimmed.ToLowerInvariant()] = $true }
        }
    }
    foreach ($paper in @($Papers)) {
        if (([string]$paper.citekey).Trim()) { continue }
        $firstAuthor = if (@($paper.authors).Count -gt 0) { [string]$paper.authors[0] } else { "Paper" }
        $surname = (@($firstAuthor -split '\s+') | Where-Object { $_ })[-1]
        $decomposed = $surname.Normalize([Text.NormalizationForm]::FormD)
        $letters = foreach ($character in $decomposed.ToCharArray()) {
            if ([Globalization.CharUnicodeInfo]::GetUnicodeCategory($character) -ne [Globalization.UnicodeCategory]::NonSpacingMark) {
                $character
            }
        }
        $surname = (-join $letters) -replace '[^A-Za-z0-9]', ''
        if (-not $surname) { $surname = "Paper" }
        $year = if (([string]$paper.year) -match '^\d{4}$') { [string]$paper.year } else { "Undated" }
        $base = "$surname$year"
        $candidate = $base
        $suffixCode = [int][char]'b'
        while ($used.ContainsKey($candidate.ToLowerInvariant())) {
            $candidate = $base + [char]$suffixCode
            $suffixCode++
        }
        $paper.citekey = $candidate
        $used[$candidate.ToLowerInvariant()] = $true
    }
    return @($Papers)
}

function Sort-CatalogPapers {
    [CmdletBinding()]
    param([object[]]$Papers)

    return @($Papers | Sort-Object `
        @{ Expression = { Get-RelevanceRank -RelevanceLevel $_.relevance_level }; Descending = $true }, `
        @{ Expression = { [int]($_.year -as [int]) }; Descending = $true }, `
        title)
}

function ConvertTo-MarkdownCell {
    [CmdletBinding()]
    param($Value)

    if ($null -eq $Value) { return "" }
    $text = [string]$Value
    $text = $text -replace "`r?`n", " "
    return $text -replace '\|', '\|'
}

function New-LiteratureCatalogMarkdown {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Papers
    )

    $lines = New-Object System.Collections.Generic.List[string]
    [void]$lines.Add("# Paper Master Index")
    [void]$lines.Add("")
    [void]$lines.Add("| Citekey | Title | Year | Relevance | Priority | Key methods | Why relevant |")
    [void]$lines.Add("|---|---|---|---|---|---|---|")

    foreach ($paper in @(Sort-CatalogPapers -Papers $Papers)) {
        $cells = @(
            (ConvertTo-MarkdownCell -Value $paper.citekey),
            (ConvertTo-MarkdownCell -Value $paper.title),
            (ConvertTo-MarkdownCell -Value $paper.year),
            (ConvertTo-MarkdownCell -Value $paper.relevance_level),
            (ConvertTo-MarkdownCell -Value $paper.recommended_priority),
            (ConvertTo-MarkdownCell -Value $paper.key_methods),
            (ConvertTo-MarkdownCell -Value $paper.why_relevant_to_current_project)
        )
        [void]$lines.Add("| $($cells -join ' | ') |")
    }

    return ($lines.ToArray() -join "`n") + "`n"
}

function ConvertTo-LiteratureCatalogJson {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Papers
    )

    $canonical = @($Papers | ForEach-Object { ConvertTo-CatalogPaper -Paper $_ })
    $ordered = @(Sort-CatalogPapers -Papers $canonical)
    return (($ordered | ConvertTo-Json -Depth 8).TrimEnd() + "`n")
}

function Assert-LiteratureCatalog {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Papers
    )

    if (@($Papers).Count -eq 0) {
        throw "Literature catalog must contain at least one paper."
    }

    $expectedFields = @(Get-LiteratureCatalogFieldNames)
    $arrayFields = @("previous_citekeys", "authors", "topic_tags", "method_tags")
    $stringFields = @($expectedFields | Where-Object { $arrayFields -notcontains $_ })
    $seenTitles = @{}
    $seenCitationKeys = @{}
    $seenDois = @{}
    $seenArxivIds = @{}

    foreach ($paper in @($Papers)) {
        $title = [string](Get-PropertyValue -Object $paper -Name "title" -Default "")
        if ([string]::IsNullOrWhiteSpace($title)) {
            throw "Every literature catalog record requires a title."
        }

        $actualFields = @($paper.PSObject.Properties.Name)
        $missingFields = @($expectedFields | Where-Object { $actualFields -notcontains $_ })
        $extraFields = @($actualFields | Where-Object { $expectedFields -notcontains $_ })
        if ($missingFields.Count -gt 0 -or $extraFields.Count -gt 0) {
            throw "Catalog record '$title' must use exactly the 35 canonical fields. Missing: $($missingFields -join ', '); extra: $($extraFields -join ', ')."
        }

        foreach ($field in $stringFields) {
            $value = $paper.PSObject.Properties[$field].Value
            if ($value -isnot [string]) {
                throw "Catalog record '$title' field '$field' must be a JSON string."
            }
        }

        foreach ($field in $arrayFields) {
            $value = $paper.PSObject.Properties[$field].Value
            if ($null -eq $value -or $value -is [string] -or $value -isnot [System.Collections.IEnumerable]) {
                throw "Catalog record '$title' field '$field' must be a JSON array of strings."
            }
            foreach ($entry in @($value)) {
                if ($entry -isnot [string]) {
                    throw "Catalog record '$title' field '$field' must contain only strings."
                }
            }
        }

        if ($paper.relevance_level -notin @("A", "B", "C", "D")) {
            throw "Catalog record '$title' has invalid relevance_level '$($paper.relevance_level)'."
        }
        if ($paper.recommended_priority -notin @("high", "medium", "background", "hold")) {
            throw "Catalog record '$title' has invalid recommended_priority '$($paper.recommended_priority)'."
        }

        $citekey = ([string]$paper.citekey).Trim()
        if ($citekey -notmatch '^[A-Za-z][A-Za-z0-9._:-]*$') {
            throw "Catalog record '$title' has invalid citekey '$citekey'."
        }
        $normalizedCitekey = $citekey.ToLowerInvariant()
        if ($seenCitationKeys.ContainsKey($normalizedCitekey)) {
            throw "Duplicate current or previous citekey in literature catalog: $citekey"
        }
        $seenCitationKeys[$normalizedCitekey] = $true
        foreach ($previousCitekey in @($paper.previous_citekeys)) {
            if ($previousCitekey -notmatch '^[A-Za-z][A-Za-z0-9._:-]*$') {
                throw "Catalog record '$title' has invalid previous citekey '$previousCitekey'."
            }
            $normalizedPrevious = ([string]$previousCitekey).ToLowerInvariant()
            if ($seenCitationKeys.ContainsKey($normalizedPrevious)) {
                throw "Duplicate current or previous citekey in literature catalog: $previousCitekey"
            }
            $seenCitationKeys[$normalizedPrevious] = $true
        }

        $localPdfPath = ([string]$paper.local_pdf_path).Trim()
        $localPdfSha256 = ([string]$paper.local_pdf_sha256).Trim()
        if ([bool]$localPdfPath -ne [bool]$localPdfSha256) {
            throw "Catalog record '$title' must set local_pdf_path and local_pdf_sha256 together."
        }
        if ($localPdfPath -and -not $localPdfPath.StartsWith("data://literature-catalog/")) {
            throw "Catalog record '$title' local_pdf_path must use data://literature-catalog/."
        }
        if ($localPdfSha256 -and $localPdfSha256 -notmatch '^[0-9a-fA-F]{64}$') {
            throw "Catalog record '$title' has invalid local_pdf_sha256."
        }

        foreach ($field in @("date_added", "last_checked")) {
            $parsedCatalogDate = [datetime]::MinValue
            if (-not [datetime]::TryParseExact(
                $paper.$field,
                "yyyy-MM-dd",
                [System.Globalization.CultureInfo]::InvariantCulture,
                [System.Globalization.DateTimeStyles]::None,
                [ref]$parsedCatalogDate
            )) {
                throw "Catalog record '$title' field '$field' must use yyyy-MM-dd format."
            }
        }

        $normalizedTitle = Get-NormalizedTitle -Title $title
        if ($seenTitles.ContainsKey($normalizedTitle)) {
            throw "Duplicate normalized title in literature catalog: $title"
        }
        $seenTitles[$normalizedTitle] = $true

        $doi = Get-NormalizedDoi -Doi ([string]$paper.doi)
        if ($doi) {
            if ($seenDois.ContainsKey($doi)) { throw "Duplicate DOI in literature catalog: $doi" }
            $seenDois[$doi] = $true
        }

        $arxivId = Get-NormalizedArxivId -ArxivId ([string]$paper.arxiv_id)
        if ($arxivId) {
            if ($seenArxivIds.ContainsKey($arxivId)) { throw "Duplicate arXiv ID in literature catalog: $arxivId" }
            $seenArxivIds[$arxivId] = $true
        }

        $adsUrl = ([string]$paper.ads_url).Trim()
        if ($adsUrl -and $adsUrl -notmatch '^https://ui\.adsabs\.harvard\.edu/abs/[^/?#]+/?$') {
            throw "Catalog record '$title' ads_url must be an ADS abstract URL; use doi/arxiv_id for other identities."
        }

        $isExplicitIdentityHold = $paper.recommended_priority -eq "hold" -and
            ([string]$paper.notes_for_current_code) -match 'UNVERIFIED_IDENTITY'
        if (-not $doi -and -not $arxivId -and [string]::IsNullOrWhiteSpace([string]$paper.ads_url) -and
            -not $isExplicitIdentityHold) {
            throw "Catalog record '$title' requires at least one DOI, arXiv ID, or ADS URL."
        }

        if ($doi -and ([string]$paper.journal) -match '(?i)^arxiv preprint$') {
            throw "Published record '$title' has a DOI but is still labelled only as an arXiv preprint."
        }
    }

    return $true
}

function Normalize-TextForComparison {
    [CmdletBinding()]
    param([AllowEmptyString()][string]$Text = "")

    $withoutBom = $Text.TrimStart([char]0xFEFF)
    return (($withoutBom -replace "`r`n", "`n") -replace "`r", "`n").TrimEnd([char[]]"`r`n")
}

function Test-TextContentEqual {
    [CmdletBinding()]
    param(
        [AllowEmptyString()][string]$Expected = "",
        [AllowEmptyString()][string]$Actual = ""
    )

    return (Normalize-TextForComparison -Text $Expected) -ceq (Normalize-TextForComparison -Text $Actual)
}

function Write-Utf8BomFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )

    $encoding = New-Object System.Text.UTF8Encoding($true)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Invoke-LiteratureCatalogUpdate {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$ConfigPath,
        [Parameter(Mandatory = $true)][string]$CatalogJsonPath,
        [Parameter(Mandatory = $true)][string]$CatalogMarkdownPath,
        [Parameter(Mandatory = $true)][string]$AsOfDate,
        [switch]$SkipLiveSearch,
        [switch]$Check
    )

    $parsedDate = [datetime]::MinValue
    if (-not [datetime]::TryParseExact($AsOfDate, "yyyy-MM-dd", [System.Globalization.CultureInfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::None, [ref]$parsedDate)) {
        throw "-AsOfDate must use yyyy-MM-dd format."
    }

    if (-not (Test-Path -LiteralPath $CatalogJsonPath -PathType Leaf)) {
        throw "Literature catalog JSON not found: $CatalogJsonPath"
    }
    if ($Check -and -not (Test-Path -LiteralPath $CatalogMarkdownPath -PathType Leaf)) {
        throw "Literature catalog Markdown not found: $CatalogMarkdownPath"
    }

    $config = Get-PaperSearchConfig -ConfigPath $ConfigPath
    $parsedCatalog = Get-Content -Raw -Encoding UTF8 -LiteralPath $CatalogJsonPath | ConvertFrom-Json
    $existingPapers = @($parsedCatalog)
    [void](Assert-LiteratureCatalog -Papers $existingPapers)
    $sourceStatus = [pscustomobject]@{
        Arxiv = [pscustomobject]@{
            Source = "arXiv"
            Candidates = @()
            SuccessfulQueries = 0
            FailedQueries = 0
            RejectedByDate = 0
            Succeeded = $false
            State = "NotRun"
        }
        Crossref = [pscustomobject]@{
            Source = "Crossref"
            Candidates = @()
            SuccessfulQueries = 0
            FailedQueries = 0
            RejectedByDate = 0
            Succeeded = $false
            State = "NotRun"
        }
    }

    if ($Check) {
        $expectedMarkdown = New-LiteratureCatalogMarkdown -Papers $existingPapers
        $storedMarkdown = Get-Content -Raw -Encoding UTF8 -LiteralPath $CatalogMarkdownPath
        if (-not (Test-TextContentEqual -Expected $expectedMarkdown -Actual $storedMarkdown)) {
            throw "Paper/catalog/papers.md is out of date with Paper/catalog/papers.json."
        }

        return [pscustomobject]@{
            Mode = "Check"
            TotalPapers = $existingPapers.Count
            AddedPapers = 0
            UpdatedFiles = @()
            LiveSearchUsed = $false
            LiveSearchSucceeded = $false
            SourceStatus = $sourceStatus
        }
    }

    $allRecords = New-Object System.Collections.Generic.List[object]
    foreach ($paper in @($existingPapers)) {
        [void]$allRecords.Add($paper)
    }

    if (-not $SkipLiveSearch) {
        $liveSearchResult = Get-LivePaperCandidates -Config $config -AsOfDate $AsOfDate
        $sourceStatus = $liveSearchResult.SourceStatus
        foreach ($candidate in @($liveSearchResult.Candidates)) {
            $scoredCandidate = ConvertTo-ScoredPaperRecord -Paper $candidate -Config $config -DefaultDate $AsOfDate
            if ($scoredCandidate.relevance_level -in @("A", "B")) {
                [void]$allRecords.Add($scoredCandidate)
            }
            else {
                Write-Verbose "Skipped low-relevance live result: $($scoredCandidate.title)"
            }
        }
    }

    $merged = @(Merge-PaperRecords -PaperRecords $allRecords.ToArray())
    $canonical = @($merged | ForEach-Object { ConvertTo-CatalogPaper -Paper $_ -DefaultDate $AsOfDate })
    $canonical = @(Set-CatalogCitekeys -Papers $canonical)
    $ordered = @(Sort-CatalogPapers -Papers $canonical)
    [void](Assert-LiteratureCatalog -Papers $ordered)

    $jsonContent = ConvertTo-LiteratureCatalogJson -Papers $ordered
    $markdownContent = New-LiteratureCatalogMarkdown -Papers $ordered
    $catalogDirectory = Split-Path -Parent $CatalogJsonPath
    if (-not (Test-Path -LiteralPath $catalogDirectory)) {
        New-Item -ItemType Directory -Path $catalogDirectory -Force | Out-Null
    }

    $updatedFiles = New-Object System.Collections.Generic.List[string]
    $storedJson = Get-Content -Raw -Encoding UTF8 -LiteralPath $CatalogJsonPath
    if (-not (Test-TextContentEqual -Expected $jsonContent -Actual $storedJson)) {
        Write-Utf8BomFile -Path $CatalogJsonPath -Content $jsonContent
        [void]$updatedFiles.Add($CatalogJsonPath)
    }

    $storedMarkdown = if (Test-Path -LiteralPath $CatalogMarkdownPath -PathType Leaf) {
        Get-Content -Raw -Encoding UTF8 -LiteralPath $CatalogMarkdownPath
    }
    else { "" }
    if (-not (Test-TextContentEqual -Expected $markdownContent -Actual $storedMarkdown)) {
        Write-Utf8BomFile -Path $CatalogMarkdownPath -Content $markdownContent
        [void]$updatedFiles.Add($CatalogMarkdownPath)
    }

    return [pscustomobject]@{
        Mode = "Update"
        TotalPapers = $ordered.Count
        AddedPapers = $ordered.Count - $existingPapers.Count
        UpdatedFiles = @($updatedFiles.ToArray())
        LiveSearchUsed = -not [bool]$SkipLiveSearch
        LiveSearchSucceeded = -not [bool]$SkipLiveSearch
        SourceStatus = $sourceStatus
    }
}

function ConvertTo-NormalizedGitPath {
    [CmdletBinding()]
    param([AllowEmptyString()][string]$Path = "")
    return $Path.Replace("\", "/").Trim("/")
}

function Get-GitLiteraturePublishPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$AsOfDate,
        [string]$RemoteName = "origin",
        [string]$BranchName = "main",
        [string[]]$ChangedPaths = @(),
        [string[]]$StagedPaths = @()
    )

    $allowedPaths = @("Paper/catalog/papers.json", "Paper/catalog/papers.md")
    $normalizedChanged = @($ChangedPaths | ForEach-Object { ConvertTo-NormalizedGitPath -Path $_ } | Where-Object { $_ } | Sort-Object -Unique)
    $normalizedStaged = @($StagedPaths | ForEach-Object { ConvertTo-NormalizedGitPath -Path $_ } | Where-Object { $_ } | Sort-Object -Unique)
    if ($normalizedStaged.Count -gt 0) {
        throw "Literature publishing requires an empty Git index. Staged paths: $($normalizedStaged -join ', ')"
    }

    $outsideScope = @($normalizedChanged | Where-Object { $allowedPaths -notcontains $_ })
    if ($outsideScope.Count -gt 0) {
        throw "Literature publishing refused because changes exist outside the two-file catalog allowlist: $($outsideScope -join ', ')"
    }

    $publishPaths = @($normalizedChanged | Where-Object { $allowedPaths -contains $_ })
    $commitMessage = "Update literature catalog for $AsOfDate"
    $commands = @()
    if ($publishPaths.Count -gt 0) {
        $commands = @(
            'git add -- "Paper/catalog/papers.json" "Paper/catalog/papers.md"',
            "git commit -m `"$commitMessage`"",
            "git push $RemoteName HEAD:$BranchName"
        )
    }

    return [pscustomobject]@{
        HasChanges = $publishPaths.Count -gt 0
        CommitMessage = $commitMessage
        Commands = $commands
        Pathspecs = $allowedPaths
        AllowlistedPaths = $publishPaths
    }
}

function Get-GitLiteratureRepositoryState {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)

    $gitRootOutput = @(& git -C $ProjectRoot rev-parse --show-toplevel 2>$null)
    if ($LASTEXITCODE -ne 0 -or $gitRootOutput.Count -eq 0) {
        throw "Literature publishing requires '$ProjectRoot' to be inside a Git repository."
    }
    $gitRoot = [string]$gitRootOutput[-1]

    $staged = @(& git -C $gitRoot diff --cached --name-only --)
    if ($LASTEXITCODE -ne 0) { throw "Unable to inspect staged Git changes." }
    $unstaged = @(& git -C $gitRoot diff --name-only --)
    if ($LASTEXITCODE -ne 0) { throw "Unable to inspect unstaged Git changes." }
    $untracked = @(& git -C $gitRoot ls-files --others --exclude-standard)
    if ($LASTEXITCODE -ne 0) { throw "Unable to inspect untracked Git files." }

    $staged = @($staged | ForEach-Object { ConvertTo-NormalizedGitPath -Path $_ } | Where-Object { $_ } | Sort-Object -Unique)
    $changed = @($staged + $unstaged + $untracked | ForEach-Object { ConvertTo-NormalizedGitPath -Path $_ } | Where-Object { $_ } | Sort-Object -Unique)
    return [pscustomobject]@{ GitRoot = $gitRoot; StagedPaths = $staged; ChangedPaths = $changed }
}

function Assert-GitLiteraturePublishReady {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [string]$RemoteName = "origin"
    )

    $state = Get-GitLiteratureRepositoryState -ProjectRoot $ProjectRoot
    [void](Get-GitLiteraturePublishPlan -AsOfDate "preflight" -ChangedPaths $state.ChangedPaths -StagedPaths $state.StagedPaths)
    $remotes = @(& git -C $state.GitRoot remote)
    if ($LASTEXITCODE -ne 0 -or $remotes -notcontains $RemoteName) {
        throw "Literature publishing requires remote '$RemoteName' to be configured."
    }
    return $state
}

function Invoke-GitLiteratureCommitPush {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$AsOfDate,
        [string]$RemoteName = "origin",
        [string]$BranchName = "main"
    )

    $state = Assert-GitLiteraturePublishReady -ProjectRoot $ProjectRoot -RemoteName $RemoteName
    $plan = Get-GitLiteraturePublishPlan -AsOfDate $AsOfDate -RemoteName $RemoteName -BranchName $BranchName -ChangedPaths $state.ChangedPaths
    if (-not $plan.HasChanges) {
        return [pscustomobject]@{ HasCommittedChanges = $false; HasPushed = $false; CommitMessage = $plan.CommitMessage }
    }

    & git -C $state.GitRoot add -- Paper/catalog/papers.json Paper/catalog/papers.md
    if ($LASTEXITCODE -ne 0) { throw "Git add failed during literature catalog publishing." }

    $stagedState = Get-GitLiteratureRepositoryState -ProjectRoot $ProjectRoot
    $unexpected = @($stagedState.StagedPaths | Where-Object { $plan.Pathspecs -notcontains $_ })
    if ($unexpected.Count -gt 0) {
        throw "Literature publishing stopped because non-allowlisted paths were staged: $($unexpected -join ', ')"
    }

    & git -C $state.GitRoot commit -m $plan.CommitMessage
    if ($LASTEXITCODE -ne 0) { throw "Git commit failed during literature catalog publishing." }
    & git -C $state.GitRoot push $RemoteName "HEAD:$BranchName"
    if ($LASTEXITCODE -ne 0) { throw "Git push failed during literature catalog publishing." }

    return [pscustomobject]@{ HasCommittedChanges = $true; HasPushed = $true; CommitMessage = $plan.CommitMessage }
}

Export-ModuleMember -Function `
    Get-LiteratureCatalogFieldNames, Resolve-LiteratureCatalogPaths, Get-PaperSearchConfig, `
    Merge-StringArray, Get-NormalizedTitle, Get-NormalizedDoi, Get-NormalizedArxivId, `
    Get-RelevanceRank, ConvertTo-ScoredPaperRecord, Merge-PaperRecords, ConvertTo-CatalogPaper, Set-CatalogCitekeys, `
    Sort-CatalogPapers, New-LiteratureCatalogMarkdown, ConvertTo-LiteratureCatalogJson, `
    Assert-LiteratureCatalog, Test-TextContentEqual, Invoke-LiteratureCatalogUpdate, `
    Get-GitLiteraturePublishPlan, Assert-GitLiteraturePublishReady, Invoke-GitLiteratureCommitPush
