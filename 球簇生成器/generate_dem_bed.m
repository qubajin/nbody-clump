function report = generate_dem_bed(cfg, templateOrCfg)
%GENERATE_DEM_BED Generate validated ph/ch inputs without running the DEM code.
%
% Required cfg fields:
%   world_size       [L W H] [m]
%   bed_height       target settled-bed height used for count estimation [m]
%   intruder_diameter spherical-intruder diameter [m]
%   output_dir       a new case directory
%
% Optional cfg fields:
%   clump_count      exact count; otherwise estimated at solid_fraction
%   solid_fraction   default 0.60
%   placement_height initial placement ceiling; default world height
%   gap_fraction     bounding-sphere gap / diameter; default 0.05
%   boundary_clearance default 0.05*minimum subsphere radius
%   seed             default 20260828
%   random_orientation default true
%
% templateOrCfg may be a ready template or a paper-style configuration
% using shape='plane', lambda, and equivalent_diameter.  All configurations
% use make_clump_template_lambda; the old overlap-only route was removed.
%
% The function writes only ph.00000000.bt and ch.00000000.bt in output_dir.
% It refuses to overwrite either file and never starts or builds the simulator.

if nargin ~= 2 || ~isstruct(cfg)
    error('generate_dem_bed:Input', 'Inputs must be cfg and template/config structs.');
end

worldSize = reshape(required_field(cfg,'world_size'),1,[]);
bedHeight = required_field(cfg,'bed_height');
intruderDiameter = required_field(cfg,'intruder_diameter');
outputDir = char(required_field(cfg,'output_dir'));
solidFraction = get_option(cfg,'solid_fraction',0.60);
placementHeight = get_option(cfg,'placement_height',worldSize(3));
gapFraction = get_option(cfg,'gap_fraction',0.05);
seed = round(get_option(cfg,'seed',20260828));
randomOrientation = logical(get_option(cfg,'random_orientation',true));

validateattributes(worldSize, {'numeric'}, {'numel',3,'real','finite','positive'});
validateattributes(bedHeight, {'numeric'}, {'scalar','real','finite','positive','<=',worldSize(3)});
validateattributes(intruderDiameter, {'numeric'}, {'scalar','real','finite','positive'});
validateattributes(placementHeight, {'numeric'}, {'scalar','real','finite','positive','<=',worldSize(3)});
validateattributes(gapFraction, {'numeric'}, {'scalar','real','finite','>=',0});
validateattributes(solidFraction, {'numeric'}, {'scalar','real','finite','>',0,'<',1});
if isempty(strtrim(outputDir))
    error('generate_dem_bed:Output', 'output_dir cannot be empty.');
end

% This call enforces the five-times spherical-intruder rule before output.
template = resolve_template(templateOrCfg);
estimate = estimate_clump_population(worldSize, bedHeight, intruderDiameter, ...
    template, solidFraction);

if isfield(cfg,'clump_count') && ~isempty(cfg.clump_count)
    clumpCount = cfg.clump_count;
    validateattributes(clumpCount, {'numeric'}, {'scalar','integer','positive'});
else
    clumpCount = estimate.estimates.ClumpCount(1);
end

nSub = template.num_spheres;
particleCount = clumpCount*nSub;
boundRadius = template.bounding_radius;
minSphereRadius = min(template.radii);
clearance = get_option(cfg,'boundary_clearance',0.05*minSphereRadius);
validateattributes(clearance, {'numeric'}, {'scalar','real','finite','>=',0});

centerLow = boundRadius + clearance;
centerHighXY = worldSize(1:2) - boundRadius - clearance;
centerHighZ = placementHeight - boundRadius - clearance;
if any(centerHighXY < centerLow) || centerHighZ < centerLow
    error('generate_dem_bed:Capacity', ...
        'Placement region is smaller than one clump bounding sphere.');
end

pitch = 2*boundRadius*(1+gapFraction);
xPositions = centered_axis_positions(centerLow, centerHighXY(1), pitch);
yPositions = centered_axis_positions(centerLow, centerHighXY(2), pitch);
zPositions = centerLow:pitch:centerHighZ;
capacity = numel(xPositions)*numel(yPositions)*numel(zPositions);
if clumpCount > capacity
    error('generate_dem_bed:Capacity', ...
        ['Requested %d clumps but safe non-overlapping placement capacity is %d. ' ...
         'Increase placement_height/world size, reduce clump size, or reduce gap_fraction.'], ...
         clumpCount, capacity);
end

oldRng = rng;
rngCleanup = onCleanup(@() rng(oldRng)); %#ok<NASGU>
rng(seed,'twister');

% Fill bottom-up.  Positions within each layer are shuffled reproducibly.
centers = zeros(clumpCount,3);
placed = 0;
[xx,yy] = ndgrid(xPositions,yPositions);
layerXY = [xx(:),yy(:)];
for k = 1:numel(zPositions)
    order = randperm(size(layerXY,1));
    take = min(size(layerXY,1),clumpCount-placed);
    if take <= 0, break; end
    rows = placed + (1:take);
    centers(rows,1:2) = layerXY(order(1:take),:);
    centers(rows,3) = zPositions(k);
    placed = placed + take;
end
if placed ~= clumpCount
    error('generate_dem_bed:Internal', 'Placement stopped at %d of %d clumps.', placed, clumpCount);
end

ch = zeros(clumpCount,18);
ph = zeros(particleCount,11);
maxTransformError = 0;
p = 0;
for c = 1:clumpCount
    if randomOrientation
        quat = random_unit_quaternion(); % [qx qy qz qw]
    else
        quat = [0 0 0 1];
    end
    Rbn = quaternion_body_to_world(quat);
    ch(c,:) = [c-1, centers(c,:), 0,0,0, quat, 0,0,0, ...
               template.principal_inertia, template.mass];

    for s = 1:nSub
        p = p + 1;
        local = template.local_centers(s,:);
        world = centers(c,:) + local*Rbn.';
        ph(p,:) = [p-1, s-1, c-1, world, local, ...
                   template.sphere_masses(s), template.radii(s)];
        reconstructed = centers(c,:) + local*Rbn.';
        maxTransformError = max(maxTransformError, norm(world-reconstructed));
    end
end

% Complete validation before opening either output file.
if size(ch,1) ~= clumpCount || size(ph,1) ~= particleCount
    error('generate_dem_bed:Validation', 'Generated row counts are inconsistent.');
end
if any(~isfinite(ch(:))) || any(~isfinite(ph(:)))
    error('generate_dem_bed:Validation', 'Generated data contains NaN or Inf.');
end
if any(ch(:,1) ~= (0:clumpCount-1).') || any(ph(:,1) ~= (0:particleCount-1).')
    error('generate_dem_bed:Validation', 'Generated IDs are not continuous from zero.');
end
if any(ph(:,3) < 0) || any(ph(:,3) >= clumpCount)
    error('generate_dem_bed:Validation', 'At least one particle has an invalid clump ID.');
end
quatNormError = max(abs(sqrt(sum(ch(:,8:11).^2,2))-1));
if quatNormError > 1e-12
    error('generate_dem_bed:Validation', 'Quaternion norm error is %.3e.', quatNormError);
end
if any(any(ch(:,15:18) <= 0)) || any(any(ph(:,10:11) <= 0))
    error('generate_dem_bed:Validation', 'Mass, radius, or inertia is not positive.');
end
lowerSurface = ph(:,4:6)-ph(:,11);
upperSurface = ph(:,4:6)+ph(:,11);
if any(any(lowerSurface < -1e-12)) || ...
   any(any(upperSurface > worldSize+1e-12))
    error('generate_dem_bed:Validation', 'At least one subsphere crosses the world boundary.');
end
minimumBoundingGap = pitch-2*boundRadius;
if minimumBoundingGap < -1e-12
    error('generate_dem_bed:Validation', 'Initial inter-clump bounding spheres overlap.');
end

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end
phPath = fullfile(outputDir,'ph.00000000.bt');
chPath = fullfile(outputDir,'ch.00000000.bt');
if exist(phPath,'file') || exist(chPath,'file')
    error('generate_dem_bed:NoOverwrite', ...
        'Refusing to overwrite existing ph/ch in %s.', outputDir);
end

try
    write_ph(phPath,ph);
    write_ch(chPath,ch);
catch exception
    if exist(phPath,'file'), delete(phPath); end
    if exist(chPath,'file'), delete(chPath); end
    rethrow(exception);
end

fprintf('\nDEM bed input generation completed (simulator was not started)\n');
fprintf('  output directory : %s\n', outputDir);
if isfield(template,'paper_label')
    fprintf('  particle / lambda: %s / %.6g\n', ...
        template.paper_label, template.nonconvexity_lambda);
    fprintf('  overlap ratio    : %.6g (= 1-lambda)\n', ...
        template.overlap_ratio);
end
fprintf('  clumps/subspheres: %d / %d\n', clumpCount, particleCount);
fprintf('  placement capacity: %d\n', capacity);
fprintf('  world size       : %.6g  %.6g  %.6g m\n', worldSize);
fprintf('  placement ceiling: %.6g m\n', placementHeight);
fprintf('  minimum bounding gap: %.6g m\n', minimumBoundingGap);
fprintf('  world.par NumberBody = %d\n', particleCount);
fprintf('  world.par ClumpNum  = %d\n', clumpCount);
fprintf('  CubeWall half size  = %.6g  %.6g  %.6g m\n', worldSize/2);
fprintf('  CubeWall center     = %.6g  %.6g  %.6g m\n', worldSize/2);

report = struct();
report.output_dir = outputDir;
report.ph_file = phPath;
report.ch_file = chPath;
report.world_size = worldSize;
report.bed_height = bedHeight;
report.placement_height = placementHeight;
report.clump_count = clumpCount;
report.particle_count = particleCount;
report.capacity = capacity;
report.minimum_bounding_gap = minimumBoundingGap;
report.maximum_transform_error = maxTransformError;
report.maximum_quaternion_norm_error = quatNormError;
report.template = template;
report.estimate = estimate;
end

function template = resolve_template(templateOrCfg)
if isstruct(templateOrCfg) && isfield(templateOrCfg,'union_volume')
    template = templateOrCfg;
    return;
end
if ~isstruct(templateOrCfg)
    error('generate_dem_bed:Template', ...
        'templateOrCfg must be a template or configuration struct.');
end

% [PREPROCESS-LINK-2025] Only the paper-consistent lambda builder remains.
template = make_clump_template_lambda(templateOrCfg);
end

function value = required_field(cfg,name)
if ~isfield(cfg,name) || isempty(cfg.(name))
    error('generate_dem_bed:Input', 'cfg.%s is required.', name);
end
value = cfg.(name);
end

function value = get_option(cfg,name,defaultValue)
if isfield(cfg,name) && ~isempty(cfg.(name))
    value = cfg.(name);
else
    value = defaultValue;
end
end

function positions = centered_axis_positions(low,high,pitch)
n = floor((high-low)/pitch)+1;
span = (n-1)*pitch;
startValue = 0.5*(low+high-span);
positions = startValue+(0:n-1)*pitch;
end

function q = random_unit_quaternion()
u = rand(1,3);
q = [sqrt(1-u(1))*sin(2*pi*u(2)), ...
     sqrt(1-u(1))*cos(2*pi*u(2)), ...
     sqrt(u(1))*sin(2*pi*u(3)), ...
     sqrt(u(1))*cos(2*pi*u(3))];
q = q/norm(q);
end

function R = quaternion_body_to_world(q)
x=q(1); y=q(2); z=q(3); w=q(4);
R = [x*x-y*y-z*z+w*w, 2*(x*y-z*w),     2*(x*z+y*w); ...
     2*(x*y+z*w),     -x*x+y*y-z*z+w*w, 2*(y*z-x*w); ...
     2*(x*z-y*w),      2*(y*z+x*w),    -x*x-y*y+z*z+w*w];
end

function write_ph(path,data)
fid = fopen(path,'w');
if fid < 0, error('generate_dem_bed:Write', 'Cannot open %s.',path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
format = ['%d %d %d ' repmat('%.12e ',1,7) '%.12e\n'];
fprintf(fid,format,data.');
end

function write_ch(path,data)
fid = fopen(path,'w');
if fid < 0, error('generate_dem_bed:Write', 'Cannot open %s.',path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
format = ['%d ' repmat('%.12e ',1,16) '%.12e\n'];
fprintf(fid,format,data.');
end
