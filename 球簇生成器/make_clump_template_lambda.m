function template = make_clump_template_lambda(cfg)
%MAKE_CLUMP_TEMPLATE_LAMBDA Build a clump using the paper's nonconvexity lambda.
%
% Required cfg fields:
%   size         specify exactly one of the following:
%     radius                fixed component-sphere radius [m]
%     equivalent_diameter   diameter of a sphere having the same union
%                           volume as the complete clump [m]
%   lambda       nonconvexity from Wang et al. (2025)
%   shape        'line', 'plane', 'tetra', 'ball', 'L', 'T',
%                'cross', or 'custom'.  'triangle' is retained as an alias
%                of the paper's plane family.
%   density      material density [kg/m^3]
%
% Optional cfg fields:
%   num_spheres                 defaults: line=2, plane=3,
%                               tetra=4, ball=1
%   custom_centers              N-by-3 coordinates for shape='custom'
%   custom_centers_are_metric   false: rescale nearest pair to requested overlap
%   monte_carlo_samples         default 1,000,000
%   seed                        default 20260828
%   plot                        default true
%
% [LAMBDA-2025] Wang et al. define lambda = Delta_d/d0.  For two equal
% neighboring component spheres, Delta_d is their center distance and
% d0=2*r.  Therefore:
%       center_distance = 2*r*lambda
%       overlap_ratio   = 1-lambda
%
% The prescribed maximum values are enforced for the requested study:
%   line-2: 1, line-3: 0.667, line-4: 0.5,
%   plane-3: 0.928, tetra-4: 0.899.
% The two Plane particles shown in Fig. 1 of Wang et al. are P3-0.5 and
% P3-0.9.  For this regular three-sphere Plane geometry, a = 1+lambda.
%
% Union volume, center of mass, and inertia are estimated with a
% deterministic Monte Carlo integration.  Overlap volume is counted once.
% For controlled comparisons, equivalent_diameter is recommended because
% the paper holds particle volume constant across configurations.

if nargin ~= 1 || ~isstruct(cfg)
    error('make_clump_template:Input', 'Input must be one configuration struct.');
end

inputShape = char(get_option(cfg, 'shape', 'line'));
shape = canonical_shape_name(inputShape);
density = get_option(cfg, 'density', []);
radius = get_option(cfg, 'radius', []);
equivalentDiameter = get_option(cfg, 'equivalent_diameter', []);
lambdaInput = get_option(cfg, 'lambda', []);
if isfield(cfg, 'overlap')
    error('make_clump_template_lambda:LegacyOverlapRemoved', ...
        ['cfg.overlap used the old overlap-fraction definition and is no ' ...
         'longer accepted. Use the paper definition cfg.lambda instead.']);
end

% [LAMBDA-2025] Use natural component counts for the three paper families.
if isfield(cfg, 'num_spheres') && ~isempty(cfg.num_spheres)
    numSpheres = cfg.num_spheres;
else
    switch lower(strtrim(shape))
        case 'plane'
            numSpheres = 3;
        case 'tetra'
            numSpheres = 4;
        case 'ball'
            numSpheres = 1;
        otherwise
            numSpheres = 2;
    end
end

sampleCount = round(get_option(cfg, 'monte_carlo_samples', 1e6));
seed = round(get_option(cfg, 'seed', 20260828));
showPlot = logical(get_option(cfg, 'plot', true));

% [EQUAL-VOLUME-2025] The paper compares shapes at identical union volume.
% Use a unit working radius when equivalent diameter is requested, then
% rescale every length/mass-inertia quantity after Monte Carlo integration.
if isempty(radius) && isempty(equivalentDiameter)
    error('make_clump_template_lambda:Size', ...
        'Specify exactly one of cfg.radius or cfg.equivalent_diameter.');
elseif ~isempty(radius) && ~isempty(equivalentDiameter)
    error('make_clump_template_lambda:Size', ...
        'Do not specify cfg.radius and cfg.equivalent_diameter together.');
elseif ~isempty(equivalentDiameter)
    validateattributes(equivalentDiameter, {'numeric'}, ...
        {'scalar','real','finite','positive'});
    radius = 1;
    useEqualVolume = true;
else
    validateattributes(radius, {'numeric'}, ...
        {'scalar','real','finite','positive'});
    useEqualVolume = false;
end
validateattributes(density, {'numeric'}, {'scalar','real','finite','positive'});
validateattributes(numSpheres, {'numeric'}, {'scalar','integer','>=',1});
validateattributes(sampleCount, {'numeric'}, {'scalar','integer','>=',10000});

% [LAMBDA-2025] Only the paper definition is accepted as an input.  The
% geometric overlap ratio is derived and reported as 1-lambda.
if isempty(lambdaInput)
    error('make_clump_template_lambda:Input', ...
        'Specify cfg.lambda using the paper definition lambda = Delta_d/d0.');
end
validateattributes(lambdaInput, {'numeric'}, ...
    {'scalar','real','finite','>=',0,'<=',1});
lambda = lambdaInput;
overlap = 1-lambda;

[lambdaMax, hasPrescribedLimit, configurationName] = ...
    maximum_nonconvexity(shape, numSpheres);
if lambda > lambdaMax + 1e-12
    error('make_clump_template_lambda:LambdaLimit', ...
        ['%s permits lambda <= %.6g for this study, but %.6g was given. ' ...
         'Increase overlap or reduce lambda.'], ...
        configurationName, lambdaMax, lambda);
end
if lambda == 0 && numSpheres > 1
    error('make_clump_template_lambda:SphereBaseline', ...
        ['lambda=0 makes all component spheres coincident and would duplicate ' ...
         'DEM contacts. Use shape=''ball'', num_spheres=1 for the sphere baseline.']);
end

centerDistance = 2 * radius * lambda;
centers = build_shape_centers(shape, numSpheres, centerDistance, cfg);
numSpheres = size(centers, 1);

% Monte Carlo integration uses a local RNG state and restores MATLAB's state.
oldRng = rng;
rngCleanup = onCleanup(@() rng(oldRng)); %#ok<NASGU>
rng(seed, 'twister');

boxLow = min(centers, [], 1) - radius;
boxHigh = max(centers, [], 1) + radius;
boxSpan = boxHigh - boxLow;
boxVolume = prod(boxSpan);
chunkSize = min(200000, sampleCount);

insideCount = 0;
sumP = zeros(1,3);
sumX2 = 0; sumY2 = 0; sumZ2 = 0;
sumXY = 0; sumXZ = 0; sumYZ = 0;

done = 0;
while done < sampleCount
    n = min(chunkSize, sampleCount - done);
    points = rand(n,3) .* boxSpan + boxLow;
    inside = false(n,1);
    for s = 1:numSpheres
        delta = points - centers(s,:);
        inside = inside | sum(delta.^2, 2) <= radius^2;
    end
    accepted = points(inside,:);
    insideCount = insideCount + size(accepted,1);
    sumP = sumP + sum(accepted,1);
    sumX2 = sumX2 + sum(accepted(:,1).^2);
    sumY2 = sumY2 + sum(accepted(:,2).^2);
    sumZ2 = sumZ2 + sum(accepted(:,3).^2);
    sumXY = sumXY + sum(accepted(:,1).*accepted(:,2));
    sumXZ = sumXZ + sum(accepted(:,1).*accepted(:,3));
    sumYZ = sumYZ + sum(accepted(:,2).*accepted(:,3));
    done = done + n;
end

if insideCount == 0
    error('make_clump_template:Integration', 'No union-volume samples were accepted.');
end

acceptedFraction = insideCount / sampleCount;
unionVolume = boxVolume * acceptedFraction;
mass = density * unionVolume;
centerOfMass = sumP / insideCount;
massWeight = density * boxVolume / sampleCount;

inertiaOrigin = massWeight * [ ...
    sumY2+sumZ2, -sumXY,       -sumXZ; ...
    -sumXY,       sumX2+sumZ2, -sumYZ; ...
    -sumXZ,       -sumYZ,       sumX2+sumY2];
c = centerOfMass(:);
inertiaCM = inertiaOrigin - mass * ((c.'*c)*eye(3) - c*c.');
inertiaCM = 0.5 * (inertiaCM + inertiaCM.');

% [EQUAL-VOLUME-2025] Similar clumps scale as V~L^3 and I~rho*L^5.
% Scaling the completed integral preserves the sampled shape and avoids a
% second Monte Carlo pass.
if useEqualVolume
    targetVolume = pi/6 * equivalentDiameter^3;
    lengthScale = (targetVolume/unionVolume)^(1/3);
    radius = radius * lengthScale;
    centerDistance = centerDistance * lengthScale;
    centers = centers * lengthScale;
    centerOfMass = centerOfMass * lengthScale;
    boxVolume = boxVolume * lengthScale^3;
    unionVolume = unionVolume * lengthScale^3;
    inertiaCM = inertiaCM * lengthScale^5;
    mass = density * unionVolume;
else
    lengthScale = 1;
end

[principalAxes, principalMatrix] = eig(inertiaCM);
[principalMoments, order] = sort(diag(principalMatrix), 'ascend');
principalAxes = principalAxes(:,order);
if det(principalAxes) < 0
    principalAxes(:,3) = -principalAxes(:,3);
end
if any(~isfinite(principalMoments)) || any(principalMoments <= 0)
    error('make_clump_template:Inertia', 'Computed principal inertia is not positive.');
end

localCenters = (centers - centerOfMass) * principalAxes;
boundingRadius = max(sqrt(sum(localCenters.^2,2)) + radius);
volumeStdError = boxVolume * sqrt(acceptedFraction*(1-acceptedFraction)/sampleCount);
relativeVolumeError = volumeStdError / unionVolume;
if relativeVolumeError > 0.01
    warning('make_clump_template:Accuracy', ...
        'Estimated relative volume error is %.3f%%; increase monte_carlo_samples.', ...
        100*relativeVolumeError);
end

template = struct();
template.shape = shape;
template.input_shape = inputShape;
template.num_spheres = numSpheres;
template.radius = radius;
template.radii = repmat(radius, numSpheres, 1);
template.overlap_ratio = overlap;
template.overlap_definition = 'overlap_ratio = 1 - nonconvexity_lambda';
template.nonconvexity_lambda = lambda;
template.lambda_max = lambdaMax;
template.lambda_limit_is_prescribed = hasPrescribedLimit;
template.configuration = configurationName;
template.paper_label = paper_particle_label(shape, numSpheres, lambda);
template.center_distance = centerDistance;
template.local_centers = localCenters;
template.union_volume = unionVolume;
template.volume_std_error = volumeStdError;
template.relative_volume_error = relativeVolumeError;
template.density = density;
template.mass = mass;
template.equivalent_sphere_diameter = (6*unionVolume/pi)^(1/3);
template.equal_volume_mode = useEqualVolume;
template.length_scale_from_integration = lengthScale;
template.sphere_masses = repmat(mass/numSpheres, numSpheres, 1);
template.principal_inertia = principalMoments(:).';
template.principal_axes_from_shape = principalAxes;
template.shape_center_of_mass = centerOfMass;
template.bounding_radius = boundingRadius;
template.aspect_ratio = particle_aspect_ratio(shape, numSpheres, lambda, ...
    localCenters, radius);
template.seed = seed;
template.monte_carlo_samples = sampleCount;

fprintf('\nClump template created\n');
fprintf('  shape / spheres       : %s / %d\n', shape, numSpheres);
fprintf('  paper label           : %s\n', template.paper_label);
fprintf('  lambda / lambda_max   : %.6g / %.6g\n', lambda, lambdaMax);
fprintf('  overlap = 1-lambda    : %.6g\n', overlap);
fprintf('  aspect ratio a        : %.6g\n', template.aspect_ratio);
fprintf('  component radius      : %.6g m\n', radius);
if useEqualVolume
    fprintf('  size control mode     : fixed equivalent volume\n');
else
    fprintf('  size control mode     : fixed component radius\n');
end
fprintf('  adjacent center gap   : %.9e m\n', centerDistance);
fprintf('  union volume          : %.9e m^3\n', unionVolume);
fprintf('  mass                  : %.9e kg\n', mass);
fprintf('  equivalent diameter   : %.9e m\n', template.equivalent_sphere_diameter);
fprintf('  principal inertia     : %.9e  %.9e  %.9e kg m^2\n', principalMoments);
fprintf('  bounding radius       : %.9e m\n', boundingRadius);
fprintf('  estimated volume error: %.4f %%\n', 100*relativeVolumeError);

if showPlot
    plot_template(localCenters, radius, shape, lambda, lambdaMax);
end
end

function centers = build_shape_centers(shape, n, d, cfg)
shapeKey = lower(strtrim(shape));
switch shapeKey
    case 'ball'
        if n ~= 1
            error('make_clump_template_lambda:Ball', ...
                'Shape ball requires num_spheres = 1.');
        end
        centers = zeros(1,3);

    case 'line'
        centers = [(0:n-1).'*d, zeros(n,2)];

    case {'plane','triangle'}
        if n ~= 3
            error('make_clump_template_lambda:Plane', ...
                'The paper''s Plane shape requires num_spheres = 3.');
        end
        % [PLANE-2025] P3-lambda: equal component spheres whose centers are
        % the vertices of an equilateral triangle in one plane.
        centers = d * [0, 0, 0; 1, 0, 0; 0.5, sqrt(3)/2, 0];

    case 'l'
        armX = ceil((n+1)/2);
        armY = n - armX;
        centers = [(0:armX-1).'*d, zeros(armX,2)];
        if armY > 0
            centers = [centers; zeros(armY,1), (1:armY).'*d, zeros(armY,1)]; %#ok<AGROW>
        end

    case 't'
        if n < 2
            error('make_clump_template:TShape', 'Shape T requires num_spheres >= 2.');
        end
        topCount = max(2, ceil(n/2));
        stemCount = n - topCount;
        topX = ((0:topCount-1) - (topCount-1)/2).'*d;
        centers = [topX, zeros(topCount,2)];
        if stemCount > 0
            centers = [centers; zeros(stemCount,1), -(1:stemCount).'*d, zeros(stemCount,1)]; %#ok<AGROW>
        end

    case 'cross'
        centers = zeros(n,3);
        directions = [1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1];
        k = 2;
        shell = 1;
        while k <= n
            for j = 1:6
                if k > n, break; end
                centers(k,:) = shell*d*directions(j,:);
                k = k + 1;
            end
            shell = shell + 1;
        end

    case 'tetra'
        if n ~= 4
            error('make_clump_template:Tetra', 'Shape tetra requires num_spheres = 4.');
        end
        centers = d/(2*sqrt(2)) * [1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1];

    case 'custom'
        if ~isfield(cfg, 'custom_centers')
            error('make_clump_template:Custom', 'cfg.custom_centers is required.');
        end
        centers = cfg.custom_centers;
        validateattributes(centers, {'numeric'}, {'2d','ncols',3,'real','finite'});
        if size(centers,1) < 1
            error('make_clump_template:Custom', 'custom_centers must contain at least one row.');
        end
        metric = logical(get_option(cfg, 'custom_centers_are_metric', false));
        if ~metric && size(centers,1) > 1
            nearest = minimum_pair_distance(centers);
            if nearest <= 0
                error('make_clump_template:Custom', 'custom_centers contains coincident points.');
            end
            centers = centers * (d/nearest);
        end

    otherwise
        error('make_clump_template:Shape', ...
            ['Unknown shape "%s". Use ball, line, plane, L, T, cross, ' ...
             'tetra, or custom.'], shape);
end
centers = centers - mean(centers,1);
end

function [lambdaMax, prescribed, name] = maximum_nonconvexity(shape, n)
% [LAMBDA-LIMITS] Values supplied for the planned configuration study.
shapeKey = lower(strtrim(shape));
prescribed = true;
switch shapeKey
    case 'ball'
        if n ~= 1
            error('make_clump_template_lambda:Configuration', ...
                'Shape ball requires num_spheres = 1.');
        end
        lambdaMax = 0;
        name = 'ball-1';

    case 'line'
        switch n
            case 2
                lambdaMax = 1.0;
            case 3
                lambdaMax = 0.667;
            case 4
                lambdaMax = 0.5;
            otherwise
                error('make_clump_template_lambda:Configuration', ...
                    ['The prescribed line-family limits are defined only for ' ...
                     '2, 3, or 4 component spheres.']);
        end
        name = sprintf('line-%d', n);

    case {'plane','triangle'}
        if n ~= 3
            error('make_clump_template_lambda:Configuration', ...
                'The paper''s Plane configuration requires 3 component spheres.');
        end
        lambdaMax = 0.928;
        name = 'plane-3';

    case 'tetra'
        if n ~= 4
            error('make_clump_template_lambda:Configuration', ...
                'The tetra configuration requires 4 component spheres.');
        end
        lambdaMax = 0.899;
        name = 'tetra-4';

    otherwise
        % Retain the original function's other geometries, but mark their
        % lambda<=1 limit as generic rather than part of the prescribed set.
        lambdaMax = 1.0;
        prescribed = false;
        name = sprintf('%s-%d', shapeKey, n);
end
end

function shape = canonical_shape_name(inputShape)
shape = lower(strtrim(inputShape));
if strcmp(shape, 'triangle')
    shape = 'plane';
end
end

function label = paper_particle_label(shape, n, lambda)
switch lower(shape)
    case 'ball'
        label = 'Ball';
    case 'line'
        label = sprintf('L%d-%.3g', n, lambda);
    case 'plane'
        label = sprintf('P%d-%.3g', n, lambda);
    case 'tetra'
        label = sprintf('B%d-%.3g', n, lambda);
    otherwise
        label = sprintf('%s%d-%.3g', upper(shape(1)), n, lambda);
end
end

function a = particle_aspect_ratio(shape, n, lambda, centers, radius)
switch lower(shape)
    case 'ball'
        a = 1;
    case 'line'
        a = 1 + (n-1)*lambda;
    case 'plane'
        % The P3 equilateral-triangle family in Fig. 1 has a=1+lambda.
        a = 1 + lambda;
    otherwise
        spans = max(centers,[],1)-min(centers,[],1)+2*radius;
        a = max(spans)/min(spans);
end
end

function value = get_option(cfg, name, defaultValue)
if isfield(cfg, name) && ~isempty(cfg.(name))
    value = cfg.(name);
else
    value = defaultValue;
end
end

function dmin = minimum_pair_distance(points)
dmin = inf;
for i = 1:size(points,1)-1
    delta = points(i+1:end,:) - points(i,:);
    dmin = min(dmin, min(sqrt(sum(delta.^2,2))));
end
end

function plot_template(centers, radius, shape, lambda, lambdaMax)
figure('Name','DEM clump template','Color','w');
hold on; axis equal; grid on; view(38,24);
[sx,sy,sz] = sphere(24);
colors = lines(size(centers,1));
for i = 1:size(centers,1)
    surf(radius*sx+centers(i,1), radius*sy+centers(i,2), ...
         radius*sz+centers(i,3), 'EdgeColor','none', ...
         'FaceColor',colors(i,:), 'FaceAlpha',0.65);
end
xlabel('principal x [m]'); ylabel('principal y [m]'); zlabel('principal z [m]');
title(sprintf('%s clump: \\lambda=%.3f, \\lambda_{max}=%.3f', ...
    shape, lambda, lambdaMax));
camlight headlight; lighting gouraud;
hold off;
end
