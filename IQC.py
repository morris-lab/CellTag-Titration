import numpy as np
import pandas as pd
from skimage import io, measure, filters, segmentation, morphology, exposure, color
import seaborn as sns

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt




# 2.2 Segmentation and quantification
from skimage.morphology import watershed
from skimage.feature import peak_local_max

from scipy import ndimage

bins = 30
class OneArea():
    #img_RGB = None
    img_R = None
    img_G = None
    img_B = None
    img_BF = None
    img_BF_rgb = None
    merged = None
    
    img_R_raw = None
    img_G_raw = None
    img_B_raw = None
    merged_raw = None
   
    data_imported = False
    
    processed_BF = {}
    processed_R = {}
    processed_G = {}
    processed_B = {}
    
       
    
    
    def __init__(self):
        pass
        
        
    ## 0. Methods for importing image ###    
    def loadImageManually(self, bright_field=None, red=None, green=None, blue=None, plot=True):
        
        self.merged = None
        
        if not bright_field is None:
            self.img_BF = bright_field.copy()
                        
        if not red is None:
            self.img_R = red.copy()
            shape_ = red.shape
            self.merged = np.zeros(shape=(shape_[0], shape_[1], 3)).astype(red.dtype)
            self.merged[:,:,0] = red
            
        if not green is None:
            self.img_G = green.copy()
            if self.merged is None:
                shape_ = green.shape
                self.merged = np.zeros(shape=(shape_[0], shape_[1], 3)).astype(green.dtype)
            self.merged[:,:,1] = green
            
        if not blue is None:
            self.img_B = blue.copy()
            if self.merged is None:
                shape_ = blue.shape
                self.merged = np.zeros(shape=(shape_[0], shape_[1], 3)).astype(blue.dtype)
            self.merged[:,:,2] = blue
        
        self.data_imported = True
        self.data_type = self.merged.dtype
        if plot:
            plt.figure()
            plt.imshow(self.merged)   
    
    def getMergedFileFromPath(self, path, plot=True):
        self.merged = io.imread(path)
        self.img_R = self.merged[:,:,0]
        self.img_G = self.merged[:,:,1]
        self.img_B = self.merged[:,:,2]
        
        self.data_type = self.merged.dtype
        self.data_imported = True 
        if plot:
            plt.figure()
            plt.imshow(self.merged)
            
            
    def getBFImageFromPath(self, path, plot=True):
        self.img_BF_rgb = io.imread(path)
        self.img_BF = color.rgb2gray(self.img_BF_rgb)
        #self.data_imported = True
        
        if plot:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(BF)
            plt.subplot(1,2,2)
            plt.imshow(self.img_BF)
                  
    def getRedImageFromPath(self, path, plot=True):
        self.img_R = io.imread(path)[:,:,0]
        
        if not self.data_imported:
            self.merged = io.imread(path)
        else:
            self.merged[:,:,0] = self.img_R
            
        self.data_type = self.merged.dtype
        self.data_imported = True
        
        if plot:
            plt.figure()
            plt.imshow(self.img_R)

            
    def getGreenImageFromPath(self, path, plot=True):
        self.img_G = io.imread(path)[:,:,1]
        

        if not self.data_imported:
            self.merged = io.imread(path)
        else:
            self.merged[:,:,1] = self.img_G
            
        self.data_type = self.merged.dtype    
        self.data_imported = True
        if plot:
            plt.figure()
            plt.imshow(self.img_G)
            
    def getBlueImageFromPath(self, path, plot=True):
        self.img_B = io.imread(path)[:,:,2]
        
        if not self.data_imported:
            self.merged = io.imread(path)
        else:
            self.merged[:,:,2] = self.img_B
            
        self.data_type = self.merged.dtype
        self.data_imported = True
        if plot:
            plt.figure()
            plt.imshow(self.img_B)
            

    ## 1. Methods for plotting ##
    
    def plotMerge(self):
        if not self.data_imported:
            print("No data imported now")
        else:
            plt.imshow(self.merged)
            
    def plotBF(self):
        if not self.img_BF_rgb is None:
            plt.imshow(self.img_BF_rgb)
        elif not self.img_BF is None:
            plt.imshow(self.img_BF)  
        else:
            print("No data imported now")
            
    def plotRed(self):
        if not self.data_imported:
            print("No data imported now")
        else:
            plt.imshow(self.img_R)
            
    def plotGreen(self):
        if not self.data_imported:
            print("No data imported now")
        else:
            plt.imshow(self.img_G)
            
    def plotBlue(self):
        if not self.data_imported:
            print("No data imported now")
        else:
            plt.imshow(self.img_B)
            
    def plotMerge_raw(self):
        if not self.data_imported:
            print("No data imported now")
        else:
            plt.imshow(self.merged_raw)
            
    def plotRed_raw(self):
        if not self.data_imported:
            print("No data imported now")
        else:
            plt.imshow(self.img_R_raw)
            
    def plotGreen_raw(self):
        if not self.data_imported:
            print("No data imported now")
        else:
            plt.imshow(self.img_G)
            
    def plotBlue_raw(self):
        if not self.data_imported:
            print("No data imported now")
        else:
            plt.imshow(self.img_B_raw)
            
    ## mehods for binalization ##
    def binarizeImageByOtsu(self, color, threshold_adjustment=0, plot=True):
        
        # read #
        if color=="Blue":
            file = self.processed_B.copy()
            img = self.img_B
        elif color == "Red":
            file = self.processed_R.copy()
            img = self.img_R
        elif color == "Green":
            file = self.processed_G.copy()
            img = self.img_G
        elif color == "BF":
            file = self.processed_BF.copy()
            img = self.img_BF
            
        
        val = filters.threshold_otsu(img)
        binary = img > val + threshold_adjustment
        
        file["binary"] = binary
        file["binarize_threshold_adjustment_otsu"] = threshold_adjustment
        file["binarize_threshold_value"] = val + threshold_adjustment
        
        # save #
        if color=="Blue":
            self.processed_B = file
        elif color=="Red":
            self.processed_R = file
        elif color=="Green":
            self.processed_G = file
        elif color=="BF":
            self.processed_BF = file
        
        if plot:
            plt.figure()
            plt.subplot(1,2,1)

            plt.title("input image")

            plt.imshow(img)
            plt.subplot(1,2,2)

            plt.title("detected area")
            plt.imshow(binary)
            
    def binarizeImageByValue(self, color, threshold_absolute_value, plot=True):
        
        # read #
        if color=="Blue":
            file = self.processed_B.copy()
            img = self.img_B
        elif color == "Red":
            file = self.processed_R.copy()
            img = self.img_R
        elif color == "Green":
            file = self.processed_G.copy()
            img = self.img_G
        elif color == "BF":
            file = self.processed_BF.copy()
            img = self.img_BF   
        
        binary = img > threshold_absolute_value
        
        file["binary"] = binary
        file["binarize_threshold_value"] = threshold_absolute_value
         
        # save #
        if color=="Blue":
            self.processed_B = file
        elif color=="Red":
            self.processed_R = file
        elif color=="Green":
            self.processed_G = file
        elif color=="BF":
            self.processed_BF = file
        
        if plot:
            plt.figure()
            plt.subplot(1,2,1)

            plt.title("input image")

            plt.imshow(img)
            plt.subplot(1,2,2)

            plt.title("detected area")
            plt.imshow(binary)
            
    ## methods for segmentation ##
    def detectObjects(self, color, segmentation_scale, plot=True):
        
        # read #
        if color=="Blue":
            file = self.processed_B.copy()
            img = self.img_B
        elif color == "Red":
            file = self.processed_R.copy()
            img = self.img_R
        elif color == "Green":
            file = self.processed_G.copy()
            img = self.img_G
        elif color == "BF":
            file = self.processed_BF.copy()
            img = self.img_BF   
        
            
        binary = file["binary"]
        
        # segmentation_get_main_info
        distance = ndimage.distance_transform_edt(binary)
        local_maxi = peak_local_max(distance, indices=False,
                                    footprint=np.ones((segmentation_scale,
                                                       segmentation_scale)),
                                    labels=binary)
        markers = morphology.label(local_maxi)
        labels_ws = watershed(-distance, markers, mask=binary)
        
        props = measure.regionprops(label_image=labels_ws,
                                    intensity_image=img)

        bboxs = np.array([prop.bbox for prop in props])
        area = np.array([prop.area for prop in props])
        cordinates = np.array([prop.centroid for prop in props])
        mean_intensity = np.array([prop.mean_intensity for prop in props])
        
        # segmentation_get_sub_info
        
        if not self.img_R is None:
            props_ = measure.regionprops(label_image=labels_ws,
                                         intensity_image=self.img_R)
            mean_intensity_ = np.array([prop.mean_intensity for prop in props_])
            file["red_mean_intensity"] = mean_intensity_
            file["red_mean_intensity_log1p"] = np.log1p(mean_intensity_)
        
        if not self.img_G is None:
            props_ = measure.regionprops(label_image=labels_ws,
                                         intensity_image=self.img_G)
            mean_intensity_ = np.array([prop.mean_intensity for prop in props_])
            file["green_mean_intensity"] = mean_intensity_
            file["green_mean_intensity_log1p"] = np.log1p(mean_intensity_)
            
        if not self.img_B is None:
            props_ = measure.regionprops(label_image=labels_ws,
                                         intensity_image=self.img_B)
            mean_intensity_ = np.array([prop.mean_intensity for prop in props_])
            file["blue_mean_intensity"] = mean_intensity_
            file["blue_mean_intensity_log1p"] = np.log1p(mean_intensity_)
            
        # update #
        file["labels_ws"] = labels_ws
        file["props"] = props
        file["bboxs"] = bboxs
        file["area"] = area
        file["cordinates"] = cordinates
        ####file["mean_intensity"] = mean_intensity
        file["cell_threshold"] = np.repeat(True, len(area)) # dummy data
        file["segmentation_scale_factor"] = segmentation_scale
        file["number_of_all_objects"] = len(props)
        
        # save #
        if color=="Blue":
            self.processed_B = file
        elif color=="Red":
            self.processed_R = file
        elif color=="Green":
            self.processed_G = file
        elif color=="BF":
            self.processed_BF = file   
        
        if plot:
            plt.figure()
  
            ax = plt.subplot(1,2,1)
            plt.title("object size")
            plt.imshow(img)
            for i in range(len(cordinates)):
                plt.text(x=cordinates[i][1], y=cordinates[i][0], s=str(area[i]),
                         fontdict= {'family': 'serif', 'color':  'white',
                                    'weight': 'normal','size': 10})
            for bb in bboxs:
                minr, minc, maxr, maxc = bb
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='white', linewidth=1)
                ax.add_patch(rect)
            
            plt.subplot(1,2,2)
            sns.distplot(area, rug=True, bins=bins)
            plt.title("cell size")

            
    def filterObjectsBySize(self, color, min_size, max_size, plot=True):
        
        # read #
        if color=="Blue":
            file = self.processed_B.copy()
            img = self.img_B
        elif color == "Red":
            file = self.processed_R.copy()
            img = self.img_R
        elif color == "Green":
            file = self.processed_G.copy()
            img = self.img_G
        elif color == "BF":
            file = self.processed_BF.copy()
            img = self.img_BF   
        
            
        bboxs = file["bboxs"]
        area = file["area"]
        cordinates = file["cordinates"]
        
        
        # process #
        cell_threshold = (area > min_size) & (area < max_size)
        
        if "cell_threshold" in file.keys():
            old_cell_threshold = file["cell_threshold"]
            cell_threshold = old_cell_threshold & cell_threshold

        bboxs_sl = bboxs[cell_threshold]
        bboxs_removed = bboxs[~cell_threshold]
        
        #update and save#
        file["cell_threshold"] = cell_threshold
        file["cell_size_filter_min"] = min_size
        file["cell_size_filter_max"] = max_size
        file["number_of_removed_object"] = len(bboxs_removed)
        file["number_of_remained_object"] = len(bboxs_sl)
        
        if color=="Blue":
            self.processed_B = file
        elif color=="Red":
            self.processed_R = file
        elif color=="Green":
            self.processed_G = file
        elif color=="BF":
            self.processed_BF = file    
            
        if plot:
            
            sns.distplot(file["area"], rug=True, bins=bins)
            plt.axvline(min_size, c="blue", label="min")
            plt.axvline(max_size, c="r", label="max")
            plt.legend()
            plt.title("distribution of detected objects")
            plt.show()

            ax = plt.subplot(1,2,1)
            plt.title("remained object: " + str(len(bboxs_sl)))
            plt.imshow(img)
            cordinates_filtered = cordinates[cell_threshold]
            area_threshold = area[cell_threshold]
                                             
            for i in range(len(cordinates_filtered)):
                plt.text(x=cordinates_filtered[i][1],
                         y=cordinates_filtered[i][0], s=str(area_threshold[i]),
                         fontdict= {'family': 'serif', 'color':  'white',
                                    'weight': 'normal','size': 10})
            for bb in bboxs_sl:
                minr, minc, maxr, maxc = bb
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='white', linewidth=1)
                ax.add_patch(rect)

            ax = plt.subplot(1,2,2)
            plt.title("removed object: " + str(len(bboxs_removed)))
            plt.imshow(img)
            
            cordinates_removed = cordinates[~cell_threshold]
            area_removed = area[~cell_threshold]
            for i in range(len(cordinates_removed)):
                plt.text(x=cordinates_removed[i][1],
                         y=cordinates_removed[i][0], s=str(area_removed[i]),
                         fontdict= {'family': 'serif', 'color':  'white',
                                    'weight': 'normal','size': 10})
            for bb in bboxs_removed:
                minr, minc, maxr, maxc = bb
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)
                
                
    def filterObjectsByCordinate(self, color, x_range_to_remove, y_range_to_remove, plot=True):
        
        # read #
        if color=="Blue":
            file = self.processed_B.copy()
            img = self.img_B
        elif color == "Red":
            file = self.processed_R.copy()
            img = self.img_R
        elif color == "Green":
            file = self.processed_G.copy()
            img = self.img_G
        elif color == "BF":
            file = self.processed_BF.copy()
            img = self.img_BF   
        
            
        bboxs = file["bboxs"]
        area = file["area"]
        cordinates = file["cordinates"]
        
        
        # process #
        cell_remove = (cordinates[:,1] >= x_range_to_remove[0]) & \
                      (cordinates[:,1] <= x_range_to_remove[1]) & \
                      (cordinates[:,0] >= y_range_to_remove[0]) & \
                      (cordinates[:,0] <= y_range_to_remove[1]) 
        cell_threshold = ~cell_remove
        
        if "cell_threshold" in file.keys():
            old_cell_threshold = file["cell_threshold"]
            cell_threshold = old_cell_threshold & cell_threshold

        bboxs_sl = bboxs[cell_threshold]
        bboxs_removed = bboxs[~cell_threshold]
        
        #update and save#
        file["cell_threshold"] = cell_threshold
        file["number_of_removed_object"] = len(bboxs_removed)
        file["number_of_remained_object"] = len(bboxs_sl)
        
        if color=="Blue":
            self.processed_B = file
        elif color=="Red":
            self.processed_R = file
        elif color=="Green":
            self.processed_G = file
        elif color=="BF":
            self.processed_BF = file    
            
        if plot:
            ax = plt.subplot(1,2,1)
            plt.title("remained: " + str(len(bboxs_sl)) + " objects")
            plt.imshow(img)
            cordinates_filtered = cordinates[cell_threshold]
            area_threshold = area[cell_threshold]
                                             
            for i in range(len(cordinates_filtered)):
                plt.text(x=cordinates_filtered[i][1],
                         y=cordinates_filtered[i][0], s=str(area_threshold[i]),
                         fontdict= {'family': 'serif', 'color':  'white',
                                    'weight': 'normal','size': 10})
            for bb in bboxs_sl:
                minr, minc, maxr, maxc = bb
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='white', linewidth=1)
                ax.add_patch(rect)

            ax = plt.subplot(1,2,2)
            plt.title("removed: " + str(len(bboxs_removed)) + " objects")
            plt.imshow(img)
            
            cordinates_removed = cordinates[~cell_threshold]
            area_removed = area[~cell_threshold]
            for i in range(len(cordinates_removed)):
                plt.text(x=cordinates_removed[i][1],
                         y=cordinates_removed[i][0], s=str(area_removed[i]),
                         fontdict= {'family': 'serif', 'color':  'white',
                                    'weight': 'normal','size': 10})
            for bb in bboxs_removed:
                minr, minc, maxr, maxc = bb
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)


    
    def plotValuesPerObjects(self, segment_color, value, scale_log=True):
      
        # read #
        if segment_color=="Blue":
            file = self.processed_B.copy()
        elif segment_color == "Red":
            file = self.processed_R.copy()
        elif segment_color == "Green":
            file = self.processed_G.copy()
        elif segment_color == "BF":
            file = self.processed_BF.copy()
               
        
        
        bboxs = file["bboxs"]
        area = file["area"]
        cordinates = file["cordinates"]
        cell_size_threshold = file["cell_threshold"]
        
        if (value in file.keys()):
            plot_value = file[value]
        else:
            print("value error")
            
        # process
        cor_filtered = cordinates[cell_size_threshold]
        plot_value = plot_value[cell_size_threshold]
        bboxs_filtered = bboxs[cell_size_threshold]
        
        
        # plot #
        plt.figure(figsize=[20,20])
        
        plt.subplot(2,2,1)
        if not type(self.merged_raw)==type(None):
            plt.imshow(self.merged_raw)
            plt.title("raw_image")
        else:
            plt.imshow(self.merged)
            plt.title("raw_image")

        ax = plt.subplot(2,2,2)
        
        
        plt.imshow(self.merged)
        plt.title(value + " in segmented object")
        for i in range(len(cor_filtered)):
            plt.text(x=cor_filtered[i][1], y=cor_filtered[i][0],
                     s=str(round(plot_value[i])),
                     fontdict= {'family': 'serif', 'color':  'white',
                                'weight': 'normal','size': 10})
        for bb in bboxs_filtered:
            minr, minc, maxr, maxc = bb
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='white', linewidth=1)
            ax.add_patch(rect)
        
        #plt.show()
        
        
        if scale_log:

            ax = plt.subplot(2,2,3)
            plt.title("quantified value per object, log scale")
            plt.scatter(x=cor_filtered[:,1], 
                        y=cor_filtered[:,0], c=np.log1p(plot_value))
            ax.set_ylim([self.merged.shape[0], 0])
            ax.set_xlim([0, self.merged.shape[1]])

            plt.subplot(2,2,4)
            #plt.xscale("log")
            sns.distplot(np.log1p(plot_value), rug=True, bins=bins)
            plt.title("histogram of " + value + ", log scale")
           
            
        else:
        
            ax = plt.subplot(2,2,3)
            plt.title("quantified value per object, log scale")
            plt.scatter(x=cor_filtered[:,1], 
                        y=cor_filtered[:,0], c=np.log1p(plot_value))
            ax.set_ylim([self.merged.shape[0], 0])
            ax.set_xlim([0, self.merged.shape[1]])

            plt.subplot(2,2,4)
            sns.distplot(plot_value, rug=True, bins=bins)
            plt.title("histogram of " + value)
            #plt.show()

        
    def getAllValuesPerObjects(self, segment_color, remove_noise=True):
        
        # read #
        if segment_color=="Blue":
            file = self.processed_B.copy()
        elif segment_color == "Red":
            file = self.processed_R.copy()
        elif segment_color == "Green":
            file = self.processed_G.copy()
        elif segment_color == "BF":
            file = self.processed_BF.copy()
            img = self.img_BF   
        
                

        # output_basic_information
        basic = {}
        basic["cordinates_y"] = file["cordinates"][:,0]
        basic["cordinates_x"] = file["cordinates"][:,1]
        basic["cell_threshold"] = file["cell_threshold"]
        basic["area"] = file["area"]
        
        
        # output_additional_information
        meta = file.copy()
        all_keys = list(file.keys())
        for i in ["binary", "labels_ws", "props", "bboxs",
                  "area","cordinates", "cell_threshold"]:
            meta.pop(i)
                   
        output = pd.concat([pd.DataFrame(basic),pd.DataFrame(meta)], axis=1)
        
        if remove_noise:
            output = output[output.cell_threshold].drop("cell_threshold", axis=1)
        return output
        
    def plotFilteringResult(self, color):
        
        # read #
        if color=="Blue":
            file = self.processed_B.copy()
            img = self.img_B
        elif color == "Red":
            file = self.processed_R.copy()
            img = self.img_R
        elif color == "Green":
            file = self.processed_G.copy()
            img = self.img_G
        elif color == "BF":
            file = self.processed_BF.copy()
            img = self.img_BF   
        
            
        cell_size_threshold = file["cell_threshold"]
        bboxs = file["bboxs"]
        area = file["area"]
        cordinates = file["cordinates"]
                                             
        bboxs_sl = bboxs[cell_size_threshold]
        cordinates_filtered = cordinates[cell_size_threshold]
        area_threshold = area[cell_size_threshold]
        
        cordinates_removed = cordinates[~cell_size_threshold]
        area_removed = area[~cell_size_threshold]
        bboxs_removed = bboxs[~cell_size_threshold]

    
        ax = plt.subplot(1,2,1)
        plt.title("remained: " + str(len(bboxs_sl)) + " objects")
        plt.imshow(img)
        cordinates_filtered = cordinates[cell_size_threshold]
        area_threshold = area[cell_size_threshold]

        for i in range(len(cordinates_filtered)):
            plt.text(x=cordinates_filtered[i][1],
                     y=cordinates_filtered[i][0], s=str(area_threshold[i]),
                     fontdict= {'family': 'serif', 'color':  'white',
                                'weight': 'normal','size': 10})
        for bb in bboxs_sl:
            minr, minc, maxr, maxc = bb
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='white', linewidth=1)
            ax.add_patch(rect)

        ax = plt.subplot(1,2,2) 
        plt.title("removed: " + str(len(bboxs_removed)) + " objects")
       
        plt.imshow(img)

        for i in range(len(cordinates_removed)):
            plt.text(x=cordinates_removed[i][1],
                     y=cordinates_removed[i][0], s=str(area_removed[i]),
                     fontdict= {'family': 'serif', 'color':  'white',
                                'weight': 'normal','size': 10})
        for bb in bboxs_removed:
            minr, minc, maxr, maxc = bb
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)



        
    ## ###

    
        


        
            

            
        
        
    

