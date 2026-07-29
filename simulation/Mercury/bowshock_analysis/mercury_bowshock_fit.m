function [fitResult, curve, selection] = mercury_bowshock_fit(crossings, allowAuto)
%MERCURY_BOWSHOCK_FIT Robust conic fit to Mercury bow-shock crossings.
%   [RESULT, CURVE] = MERCURY_BOWSHOCK_FIT(CROSSINGS) fits only rows whose
%   Status is "Confirmed".  Set ALLOWAUTO true to create the provisional
%   first-pass fit used before manual review.

if nargin < 2
    allowAuto = false;
end

required = ["X_ab_RM","Y_ab_RM","Z_MSO_RM","Status"];
assert(all(ismember(required, string(crossings.Properties.VariableNames))), ...
    'MercuryBowShock:FitColumns', 'The crossing table is missing fit columns.');

status = string(crossings.Status);
use = status == "Confirmed";
mode = "Confirmed";
if nnz(use) < 3 && allowAuto
    use = status == "Confirmed" | status == "Auto";
    mode = "ProvisionalAuto";
end
assert(nnz(use) >= 3, 'MercuryBowShock:TooFewConfirmed', ...
    'Confirm at least three crossings before the final fit.');

inputMask = use;
[use,selection] = selectOutermost(crossings,inputMask);
x = crossings.X_ab_RM(use);
rho = hypot(crossings.Y_ab_RM(use), crossings.Z_MSO_RM(use) - 0.196);
finiteRows = isfinite(x) & isfinite(rho);
x = x(finiteRows);
rho = rho(finiteRows);
assert(numel(x) >= 3, 'MercuryBowShock:TooFewFinite', ...
    'At least three finite crossings are required.');

p0 = [0.5, 1.04, 2.86];
options = optimset('Display','off','MaxFunEvals',30000, ...
    'MaxIter',10000,'TolX',1e-10,'TolFun',1e-10);
p = fminsearch(@(q) objective(q,x,rho), p0, options);

residual = implicitResidual(p,x,rho);
rmse = sqrt(mean(residual.^2));
madResidual = 1.4826 * median(abs(residual - median(residual)));
nose = p(1) + p(3)/(1 + p(2));

fitResult = table(p(1),p(2),p(3),nose,rmse,rmse*2440, ...
    madResidual,nnz(inputMask),numel(x),nnz(inputMask)-numel(x),mode, ...
    'VariableNames',{'X0','epsilon','L','L_SSP_RM','RMSE_RM', ...
    'RMSE_km','RobustScatter_RM','NInputPoints','NPoints', ...
    'NDiscardedInner','Mode'});

theta = linspace(-2.55,2.55,1200);
r = p(3)./(1 + p(2).*cos(theta));
xCurve = p(1) + r.*cos(theta);
rhoCurve = abs(r.*sin(theta));
valid = isfinite(xCurve) & isfinite(rhoCurve) & ...
    xCurve > -6 & xCurve < 4 & rhoCurve < 8;
curve = table(xCurve(valid).',rhoCurve(valid).', ...
    'VariableNames',{'X_ab_RM','rho_RM'});
end

function [keep,selection] = selectOutermost(crossings,inputMask)
% Consecutive same-direction points separated by <=2 h belong to one leg.
% The point with the greatest MSO radial distance is the outermost.
keep = false(height(crossings),1);
reason = repmat("Not eligible",height(crossings),1);
groupID = zeros(height(crossings),1);
radius = sqrt(crossings.X_ab_RM.^2 + crossings.Y_ab_RM.^2 + ...
    crossings.Z_MSO_RM.^2);
nextGroup = 0;

for direction = ["Inbound","Outbound"]
    rows = find(inputMask & string(crossings.Direction)==direction);
    if isempty(rows), continue; end
    [~,order] = sort(crossings.UTC(rows));
    rows = rows(order);
    groupStart = 1;
    while groupStart <= numel(rows)
        groupEnd = groupStart;
        while groupEnd < numel(rows) && ...
                hours(crossings.UTC(rows(groupEnd+1)) - ...
                crossings.UTC(rows(groupEnd))) <= 2
            groupEnd = groupEnd + 1;
        end
        members = rows(groupStart:groupEnd);
        nextGroup = nextGroup + 1;
        [~,outerPos] = max(radius(members));
        chosen = members(outerPos);
        keep(chosen) = true;
        groupID(members) = nextGroup;
        reason(members) = "Inner duplicate on same orbital leg";
        reason(chosen) = "Outermost point retained";
        groupStart = groupEnd + 1;
    end
end

selection = table(crossings.ID,inputMask,groupID,radius,keep,reason, ...
    'VariableNames',{'ID','EligibleInput','OrbitalLegGroup', ...
    'RadialDistance_RM','SelectedForFit','SelectionReason'});
selection = selection(inputMask,:);
end

function value = objective(p,x,rho)
% Huber loss plus soft physical bounds keeps fminsearch toolbox-free.
if any(~isfinite(p))
    value = realmax('double')/100;
    return
end
r = implicitResidual(p,x,rho);
delta = 0.08;
a = abs(r);
loss = sum(0.5*(a <= delta).*a.^2 + ...
    (a > delta).*delta.*(a - 0.5*delta));

lower = [-0.75, 0.35, 1.0];
upper = [ 1.50, 1.80, 5.0];
below = max(lower-p,0);
above = max(p-upper,0);
penalty = 1e5*sum(below.^2 + above.^2);
value = loss + penalty;
end

function r = implicitResidual(p,x,rho)
r = hypot(x-p(1),rho) + p(2).*(x-p(1)) - p(3);
end
