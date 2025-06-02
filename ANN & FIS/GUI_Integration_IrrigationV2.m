clear all;
clc;
close all;
warning off;

% Load FIS and ANN
FIS_path = 'D:\BACHELOR OF MECHATRONICS ENGINEERING WITH HONOURS\Year 4\Y4S3\Intelligent Systems\Mini Project\FIS_irrigationV6.fis';
fis = readfis(FIS_path); 
load('water_quality_model.mat', 'net');

% Initialize output structure
output = struct();

% Create a timer for real-time execution
monitoringTimer = timer(...
    'ExecutionMode', 'fixedRate', ...  
    'Period', 1, ...                  
    'TimerFcn', @(~,~) runIrrigationSystem(fis, net), ...  
    'StopFcn', @(~,~) disp('Monitoring stopped.') ...
);

% Start the GUI menu
keepRunning = true;
while keepRunning
    k = menu('Intelligent Irrigation System', ...
             'Train ANN', ...
             'Start Real-Time Monitoring', ...
             'Stop Monitoring', ...
             'Close Program');
    
    switch k
        case 1  
            trainANN();
            
        case 2  
            if strcmp(monitoringTimer.Running, 'off')
                start(monitoringTimer);
                disp('Real-time monitoring started.');
            else
                disp('Monitoring is already running.');
            end
            
        case 3  
            if strcmp(monitoringTimer.Running, 'on')
                stop(monitoringTimer);
                disp('Monitoring stopped.');
            else
                disp('Monitoring was not running.');
            end
            
        case 4  
            if strcmp(monitoringTimer.Running, 'on')
                stop(monitoringTimer);
            end
            delete(monitoringTimer);
            keepRunning = false;
            close all;
            disp('Program closed.');
    end
end

% --- Helper Functions ---
function runIrrigationSystem(fis, net)
    fid = fopen('Irrigation_input_dataV2.txt', 'r'); 
    raw = fread(fid, inf, '*char')'; 
    fclose(fid); 
   
    input_data = jsondecode(raw);

    disp(input_data);
    
    soil_moisture = input_data.soil_moisture;
    light_intensity = input_data.light_intensity;
    air_humidity = input_data.air_humidity;    
    air_temperature = input_data.air_temperature;  
    input_vector = [soil_moisture, light_intensity, air_humidity, air_temperature];
    FIS_output = evalfis(fis, input_vector);
    disp('FIS Output:');
    disp(FIS_output);
    disp(FIS_output(1));
    disp(FIS_output(2));
    
    pH = input_data.pH;
    Hardness = input_data.Hardness; 
    Solids = input_data.Solids; 
    Chloramines = input_data.Chloramines; 
    Conductivity = input_data.Conductivity; 
    Organic_carbon = input_data.Organic_carbon; 
    Turbidity = input_data.Turbidity; 
    Sulfate = input_data.Sulfate;
    Trihalomethanes = input_data.Trihalomethanes;
    ANN_input = [pH, Hardness, Solids, Chloramines, ...
                Conductivity, Organic_carbon, Turbidity, Sulfate, Trihalomethanes]';
    % Check input size before passing to the network
    if length(ANN_input) ~= net.inputs{1}.size
        error('Input vector size does not match the trained network.');
    end
    ANN_output = net(ANN_input);
    disp('Predicted Output:');
    disp(ANN_output);
    
    % Generate Result of ANN and FIS to Text File
    output.voltage = FIS_output(1)
    output.duration = FIS_output(2)
    output.water_quality = ANN_output
    jsonOutput = jsonencode(output);
    fileID = fopen('Irrigation_output_data.txt', 'w'); 
    fwrite(fileID, jsonOutput, 'char');
    fclose(fileID);
end

function trainANN()
    % Load input data from CSV file
    Input_data = readmatrix('input2test.xlsx'); 
    output_data = readmatrix('output2test.xlsx');  
    
    x = Input_data';
    t = output_data';
    
    if any(isnan(x(:))) || any(isinf(x(:)))
        x(isnan(x) | isinf(x)) = 0;
    end
    
    % Normalize input data - critical for neural networks
    [x_norm, ps] = mapstd(x);
    
    % Choose a more powerful Training Function
    trainFcn = 'trainlm';  % Levenberg-Marquardt algorithm
    
    % Create a Pattern Recognition Network with optimized architecture
    hiddenLayerSize = [35 25];  
    net = patternnet(hiddenLayerSize, trainFcn);
    
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 75/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 10/100;
    
    % Adjust Training Parameters for better convergence
    net.trainParam.epochs = 1000;  
    net.trainParam.max_fail = 20;  
    net.trainParam.min_grad = 1e-8;  
    net.performParam.regularization = 0.1;  
    
    % Train the Network
    [net, tr] = train(net, x_norm, t);
    
    % Test the Network
    y = net(x_norm);
    e = gsubtract(t, y);
    performance = perform(net, t, y);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind) / numel(tind);
    
    % Compute Regression (R) for Training, Validation, and Testing
    trainTargets = t(:, tr.trainInd);
    trainOutputs = y(:, tr.trainInd);
    [R_train, ~] = corrcoef(vec2ind(trainTargets), vec2ind(trainOutputs));
    
    valTargets = t(:, tr.valInd);
    valOutputs = y(:, tr.valInd);
    [R_val, ~] = corrcoef(vec2ind(valTargets), vec2ind(valOutputs));
    
    testTargets = t(:, tr.testInd);
    testOutputs = y(:, tr.testInd);
    [R_test, ~] = corrcoef(vec2ind(testTargets), vec2ind(testOutputs));
    
    % Display Regression (R) Values
    disp(['Training Regression (R): ', num2str(R_train(1,2))]);
    disp(['Validation Regression (R): ', num2str(R_val(1,2))]);
    disp(['Testing Regression (R): ', num2str(R_test(1,2))]);
    
    % Display overall accuracy (additional line)
    disp(['Overall accuracy: ', num2str((1-percentErrors)*100), '%']);
    
    % Plot Regression Graphs
    figure, plotregression(trainTargets, trainOutputs, 'Training')
    figure, plotregression(valTargets, valOutputs, 'Validation')
    figure, plotregression(testTargets, testOutputs, 'Testing')
    figure, plotregression(t, y, 'Overall')
    
    % View the Network
    view(net);
    
    % Plots
    %figure, plotperform(tr)
    %figure, plottrainstate(tr)
    %figure, ploterrhist(e)
    %figure, plotconfusion(t, y)
    %figure, plotroc(t, y)
    
    % Save the trained model (additional line)
    save('water_quality_model.mat', 'net', 'ps');
    disp('ANN training complete.');
end