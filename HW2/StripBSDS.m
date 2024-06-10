names = dir('*.mat');
for ii = 1:50
    load(names(ii).name);
    c = strsplit(names(ii).name,'.');
    name = c{1};
    for jj = 1:5
        seg = groundTruth{jj}.Segmentation;
        map = groundTruth{jj}.Boundaries;
        imwrite(seg,[name,'_s',num2str(jj),'.tif']);
        imwrite(map,[name,'_e',num2str(jj),'.bmp']);
    end
end
        