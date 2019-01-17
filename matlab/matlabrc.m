%MATLABRC Master startup MATLAB script.
%   MATLABRC is automatically executed by MATLAB during startup.
%   It establishes the MATLAB path, sets the default figure size,
%   and sets a few uicontrol defaults.
%
%   On multi-user or networked systems, the system manager can put
%   any messages, definitions, etc. that apply to all users here.
%
%   MATLABRC also invokes a STARTUP command if the file 'startup.m'
%   exists on the MATLAB path.
 
%   Copyright 1984-2015 The MathWorks, Inc.

if isdeployed || ismcc
    % Turn off warnings about built-in not being visible.
    warning off MATLAB:predictorNoBuiltinVisible
end

if ismcc || ~isdeployed
    % Try to catch a potential search path issue if PATHDEF.M throws an error
    % or when USEJAVA.M is called. USEJAVA is not a builtin and only builtins
    % are guaranteed to be available during initialization.

    try
        % Set up path.
        oldPath = matlabpath;

        % We check for a RESTOREDEFAULTPATH_EXECUTED variable to check whether
        % RESTOREDEFAULTPATH was run. If it was, we don't want to use PATHDEF,
        % since it may have been the culprit of the faulty path requiring us to
        % recover using RESTOREDEFAULTPATH.
        if exist('pathdef','file') && ~exist('RESTOREDEFAULTPATH_EXECUTED','var')
            matlabpath(pathdef);
        end

        % Avoid running directly out of the bin/arch directory as this is
        % not supported.
        if ispc,
            pathToBin = [matlabroot,filesep,'bin',filesep,computer('arch')];
            if isequal(pwd, pathToBin),
                cd (matlabroot);
            end;
        end;

        % Display helpful hints.
        % If the MATLAB Desktop is not running, then use the old message, since
        % the Help menu will be unavailable.
        if ~ismcc
            if ~usejava('Desktop')
                disp(' ')
                if usejava('jvm')
                    disp(getString(message('MATLAB:matlabrc:ToGetStartedMessage')))
                else
                    disp(getString(message('MATLAB:matlabrc:ForOnlineDocumentation')))
                end
                disp(getString(message('MATLAB:matlabrc:ProductInformationMessage')))
                disp(' ')
            end
        end

        % Add installed custom Toolboxes to the path.
        matlab.internal.toolboxes.addInstalledToolboxesToPath;
        
        % Add installed zip-type add-ons to the path.
        matlab.internal.zipAddOns.addInstalledZipAddOnsToPath;
    
    catch exc
        %Show the error that occurred, in case that helps:
        disp(exc.message);
        % When modifying this code, you can only use builtins
        warning(message('MATLAB:matlabrc:SuspectedPathProblem'));
        % The initial path was $MATLAB/toolbox/local, so ensure we still have it
        if strncmp(computer,'PC',2)
            osPathsep = ';';
        else
            osPathsep = ':';
        end
        matlabpath([oldPath osPathsep matlabpath])
    end
end

% Initialize MATLAB Graphics
matlab.graphics.internal.initialize;

try
    % The RecursionLimit forces MATLAB to throw an error when the specified
    % function call depth is hit.  This protects you from blowing your stack
    % frame (which can cause MATLAB and/or your computer to crash).  
    % The default is set to 500.
    % Uncomment the line below to set the recursion limit to something else.
    % Set the value to inf if you don't want this protection
    % set(0,'RecursionLimit',700)
catch exc
    warning(message('MATLAB:matlabrc:RecursionLimit', exc.identifier, exc.message));
end

% Set default warning level to WARNING BACKTRACE.  See help warning.
warning backtrace

% Do not initialize the desktop or the preferences panels for deployed
% applications, which have no desktop.
% Do not initialize default profiler filters since they are not deployable
% either.
if ~isdeployed && ~ismcc
    try
        % For the 'edit' command, to use an editor defined in the $EDITOR
        % environment variable, the following line should be uncommented
        % (UNIX only)

        %system_dependent('builtinEditor','off')

        if usejava('mwt')
            initprefs %% init java prefs system if java is present
            initdesktoputils  %% init desktop setup code if java is present
        end
    catch exc
        warning(message('MATLAB:matlabrc:InitJava', exc.identifier, exc.message));
    end
	% add default profiler filters
    try
		files = { 'profile.m', 'profview.m', 'profsave.m', 'profreport.m', 'profviewgateway.m' };
        for i = 1:length(files)
			fname = which(files{i});
			% if we can't find the profiler files on the path, try the
			% "default" location.
			if strcmp(fname, '')
				fname = fullfile(matlabroot,'toolbox','matlab','codetools',files{i});
			end
			callstats('pffilter', 'add', fname);
        end
    catch exc
		warning(message('MATLAB:matlabrc:ProfilerFilters'));
    end

    try
        % Enable the device plugin detection manager.
        pl = internal.deviceplugindetection.Manager.getInstance();
    catch
    end
end

try
    % Text-based preferences
    NumericFormat = system_dependent('getpref','GeneralNumFormat2');
    % if numeric format is empty, check the old (pre-R14sp2) preference
    if (isempty(NumericFormat))
        NumericFormat = system_dependent('getpref','GeneralNumFormat');
    end
    if ~isempty(NumericFormat)
        eval(['format ' NumericFormat(2:end)]);
    end
    NumericDisplay = system_dependent('getpref','GeneralNumDisplay');
    if ~isempty(NumericDisplay)
        format(NumericDisplay(2:end));
    end
    if (strcmp(system_dependent('getpref','GeneralEightyColumns'),'Btrue'))
        feature('EightyColumns',1);
    end

catch exc
    warning(message('MATLAB:matlabrc:InitPreferences', exc.identifier, exc.message));
end


try
    % Enable/Disable selected warnings by default
    warning on  MATLAB:namelengthmaxExceeded
    warning off MATLAB:mir_warning_unrecognized_pragma
    warning off MATLAB:COPYFILE:SHFileOperationErrorID
    warning off MATLAB:subscripting:noSubscriptsSpecified %future incompatibity

    if ismcc
        warning off MATLAB:dispatcher:nameConflict
    end

    warning off MATLAB:JavaComponentThreading
    warning off MATLAB:JavaEDTAutoDelegation

    % Random number generator warnings
    warning off MATLAB:RandStream:ReadingInactiveLegacyGeneratorState
    warning off MATLAB:RandStream:ActivatingLegacyGenerators

    % Debugger breakpoint suppressed by Desktop
    warning off MATLAB:Debugger:BreakpointSuppressed

    warning off MATLAB:class:DynPropDuplicatesMethod

    warning off MATLAB:lang:badlyScopedReturnValue
    
catch exc
    warning(message('MATLAB:matlabrc:DisableWarnings', exc.identifier, exc.message));
end
% Initial Working folder 

setInitialWorkingFolder();

% Clean up workspace.
clear
 

% We don't run startup.m from here in deployed apps--it's run from mclmcr.cpp
% because deployed apps call javaaddpath after running matlabrc.m, which clears
% the global workspace.
if ismcc || ~isdeployed
    try
        % Execute startup MATLAB script, if it exists.
        if (exist('startup','file') == 2) ||...
                (exist('startup','file') == 6)
            startup
        end
    catch exc
        warning(message('MATLAB:matlabrc:Startup', exc.identifier, exc.message));
    end
end

% Defer echo until startup is complete
if strcmpi(system_dependent('getpref','GeneralEchoOn'),'BTrue')
    echo on
end
% Run deployment configuration file only in deployed mode.
if( isdeployed && ~ismcc && exist('deployrc', 'file'))
    deployrc;
end

cd C:\Users\Administrator\Desktop\project\matlab;
