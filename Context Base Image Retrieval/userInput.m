% Clear workspace
clear

filePath = fullfile(pwd, "queryImages");  % Get query image folder being used


% Get access to the database, should be in same folder as data
database = fullfile(pwd, "processed");   %  Same folder should have the data under name "processed"

% In the case unable to find the classifier model
if not(isfile(fullfile(pwd, "classifier.mat")))
    fprintf("Unable to load Classifier Model. Please Run bagOfFeature.mlx file first\n");
        
    % Potentially ask if user wants to run the file, then run it here
    prompt = "Would you like to run bagOfFeature.mlx? Y/N:\n";    
    txt = input(prompt, "s");
    
    if strcmp(txt, "Y") % If yes then run the file
        fprintf("Running bagOfFeature...");

        if not(isfile(fullfile(pwd, "bagOfFeature.mlx")))  % If file not available
            fprintf("Cannot find bagOfFeature.mlx, ensure it's in the same folder as this file\nStopping execution of script");
            return;
        end

        run("bagOfFeature.mlx");  % Run and train the Classifier model, will take around 10 mins (at least on my machine)

        fprintf("Done Training Model")
    elseif strcmp(txt, 'N')  % If No
        fprintf("Stopping execution of script")
        return;
    else  % Else if invalid input
        fprintf("Invalid Input, ensure input is either Y or N\nStopping Execution of script");
        return;
    end
end

if not(isfile(fullfile(pwd, "classifier.mat")))  % If unable to find file again
    fprintf("Error in building Model, please try building the model separately by running the 'bagOfFeature.mlx' and running this script again.");
    return;
end

load classifier.mat % Load classifier



fprintf("Query Images should be placed in the stated folder: %s\n", filePath);

%  Get user prompt
prompt = "Welcome to our implementation of CBIR!\nPlease place your query image in the folder listed above this prompt\nInput Query Image File (Include file format, e.g. image01.png):";
txt = input(prompt, "s");

% Check if input given is valid
imgPath = fullfile(filePath, txt);
while not(isfile(imgPath))
    prompt = "Unable to find Image specified. Please Try again and make sure image is in the folder.\nImage File (Include file format):";
    txt = input(prompt,"s");

    imgPath = fullfile(filePath, txt);
end

% If File exists
fprintf("Found Image, attempting to query...\n")
fprintf("Image Path: %s\n", imgPath)

% Load Original image and preprocess original iamge
original = imread(imgPath);

imds = imageDatastore(imgPath);  % Create imageDatastore of query Image
imds.ReadFcn = @preprocess;  % Define how to read query image

[labelIdx, score ] = predict(classifier, imds, "Verbose", false);  % Predict Category

result = classifier.Labels(labelIdx);  % Get category label

% Get image from database
output = fullfile(database, result);
fprintf("Result label is : %s\n", output);

rows = 2;
columns = 3;
figure, subplot(rows, columns, 1), imshow(original), title('Query Image');
for i = 1:5
    img = imread(fullfile(output, result+ "_" + i + ".png"));
    subplot(rows, columns, i+1), imshow(img), title("Output Image: " + i);
end


% Unsure if this to be used at or not
function newI = preprocess(I)  % Preprocessing the image before passing to get extracted

    [height,width,numChannels] = size(I);

    I = imread(I);
       
    if numChannels > 1
        grayImage = rgb2gray(I);
    else
        grayImage = I;
    end

    grayImage = imresize(grayImage,[512 512]);
    
    %smoothing
    grayImage = medfilt2(grayImage);
    
    %contrast stretching
    grayImage = imadjust(grayImage,stretchlim(grayImage),[0.05 0.95]); % ~25% accuracy
    
    %sharpening
    %grayImage = unsharpFilter(grayImage);

    %Edge only
    grayImage = edge(grayImage, "canny");


    newI = grayImage;
end