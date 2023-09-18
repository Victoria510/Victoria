    function Iout = readAndPreprocessImage(filename)
  

        I = imread(filename); 
        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image.
        if ismatrix(I) 
            I = cat(3,I,I,I); 
         
        Iout = imresize(I, [227 227]); 

          end