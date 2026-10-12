function result = estimate_clump_population(worldSize, bedHeight, intruderDiameter, templateOrCfg, solidFractions)
%ESTIMATE_CLUMP_POPULATION Estimate clumps required for a settled bed.
%
% result = estimate_clump_population(worldSize, bedHeight, intruderDiameter, ...
%                                    templateOrCfg, solidFractions)
%
% Inputs:
%   worldSize      [L W H] in m
%   bedHeight      target settled-bed height in m
%   intruderDiameter spherical-intruder diameter [m]; use [] to report only
%                    the maximum allowable sphere diameter
%   templateOrCfg  template result, or a paper-lambda configuration for
%                  make_clump_template_lambda
%   solidFractions optional vector; default [0.55 0.60 0.64]
%
% Mandatory boundary rule:
%   every world dimension >= 5*intruderDiameter.

if nargin < 4
    error('estimate_clump_population:Input', 'At least four inputs are required.');
end
if nargin < 5 || isempty(solidFractions)
    solidFractions = [0.55 0.60 0.64];
end

worldSize = reshape(worldSize,1,[]);
validateattributes(worldSize, {'numeric'}, {'numel',3,'real','finite','positive'});
validateattributes(bedHeight, {'numeric'}, {'scalar','real','finite','positive'});
validateattributes(solidFractions, {'numeric'}, {'vector','real','finite','>',0,'<',1});
if bedHeight > worldSize(3)
    error('estimate_clump_population:BedHeight', ...
        'bedHeight %.6g m exceeds world height %.6g m.', bedHeight, worldSize(3));
end

[sphereDiameter, dimensionRatios, maximumAllowedSphere] = ...
    check_five_times_rule(worldSize, intruderDiameter);

template = resolve_template(templateOrCfg);
validate_template(template);

bedVolume = worldSize(1) * worldSize(2) * bedHeight;
solidFractions = solidFractions(:);
clumpCounts = ceil(solidFractions * bedVolume / template.union_volume);
subsphereCounts = clumpCounts * template.num_spheres;
totalMasses = clumpCounts * template.mass;
porosities = 1 - solidFractions;

estimateTable = table(solidFractions, porosities, clumpCounts, ...
    subsphereCounts, totalMasses, ...
    'VariableNames', {'SolidFraction','Porosity','ClumpCount','SubsphereCount','TotalMass_kg'});

fprintf('\nFive-times boundary check\n');
fprintf('  world [L W H]           : %.6g  %.6g  %.6g m\n', worldSize);
fprintf('  max allowed sphere diam.: %.6g m\n', maximumAllowedSphere);
if ~isempty(intruderDiameter)
    fprintf('  sphere intruder diameter: %.6g m\n', sphereDiameter);
    fprintf('  world/diameter ratios   : %.3f  %.3f  %.3f\n', dimensionRatios);
end
fprintf('  target bed height/volume: %.6g m / %.9g m^3\n', bedHeight, bedVolume);
fprintf('  clump union volume      : %.9e m^3\n\n', template.union_volume);
if isfield(template,'paper_label')
    fprintf('  paper particle label    : %s\n', template.paper_label);
    fprintf('  lambda / overlap ratio  : %.6g / %.6g\n\n', ...
        template.nonconvexity_lambda, template.overlap_ratio);
end
disp(estimateTable);

result = struct();
result.world_size = worldSize;
result.bed_height = bedHeight;
result.bed_volume = bedVolume;
result.intruder_diameter = sphereDiameter;
result.maximum_allowed_intruder_diameter = maximumAllowedSphere;
result.dimension_ratios = dimensionRatios;
result.template = template;
result.estimates = estimateTable;
end

function template = resolve_template(templateOrCfg)
if isstruct(templateOrCfg) && isfield(templateOrCfg,'union_volume')
    template = templateOrCfg;
    return;
end
if ~isstruct(templateOrCfg)
    error('estimate_clump_population:Template', ...
        'templateOrCfg must be a template or configuration struct.');
end

% [PREPROCESS-LINK-2025] The legacy overlap-only builder was removed.  Every
% configuration now follows the paper's lambda definition.
template = make_clump_template_lambda(templateOrCfg);
end

function [sphereDiameter, ratios, maxAllowed] = check_five_times_rule(worldSize, intruderDiameter)
maxAllowed = min(worldSize)/5;
sphereDiameter = [];
ratios = [];
if isempty(intruderDiameter)
    return;
end
validateattributes(intruderDiameter, {'numeric'}, {'scalar','real','finite','positive'});
sphereDiameter = intruderDiameter;
ratios = worldSize/sphereDiameter;
if any(ratios < 5-1e-12)
    required = 5*sphereDiameter;
    error('estimate_clump_population:BoundaryEffect', ...
        ['Five-times rule failed. Sphere intruder diameter is %.6g m; ' ...
         'each world dimension must be at least %.6g m. Current ratios: %.3f %.3f %.3f.'], ...
         sphereDiameter, required, ratios);
end
end

function validate_template(template)
required = {'union_volume','num_spheres','mass','bounding_radius','local_centers', ...
            'radii','sphere_masses','principal_inertia'};
for i = 1:numel(required)
    if ~isfield(template, required{i})
        error('estimate_clump_population:Template', ...
            'Template is missing field %s.', required{i});
    end
end
if ~isfinite(template.union_volume) || template.union_volume <= 0
    error('estimate_clump_population:Template', 'Template union volume must be positive.');
end
end
