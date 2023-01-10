import cv2
import os
import random
from PIL import Image
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import datetime
import tensorflow as tf

class Preprocess:
    def __init__(self, directory, frameDirectory ="", isVideo=True):
        self.directory = directory
        self.frameDirectory = frameDirectory
        self.isVideo = isVideo

    
    def plot_album(self, figsz, image_paths, nrows, ncols, titles, fig_title = None, resz = (-1,-1)):
        fig, axes = plt.subplots(nrows, ncols,figsize = figsz)
        # this assumes the images are in images_dir/album_name/<name>.jpg
        for imp, ax , title in zip(image_paths, axes.ravel(),titles):
            if resz == (-1,-1):
                img = mpimg.imread(imp)
            else:
                img = mpimg.imread(imp).copy()
                img = cv2.resize(img,resz)
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
            #fig.tight_layout()

        if fig_title != None:
            fig.suptitle(fig_title, fontsize=16)
            fig.tight_layout()
        return


    #Return Selected Videos : Dimensions, Duration, Framerate and print it
    def get_video_specs(self, vid, toPrint = True):
        if(not self.isVideo):
            print("This directory doesn't contain videos")
            return

        video = cv2.VideoCapture(vid)

        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        width = video.get(3)
        height = video.get(4)
        if toPrint:
            print(f"The video has size of ({width},{height}) , duration of: {frame_count/fps} sec FPS: {fps} and total frames: {frame_count}.")
        
        return ((width,height),fps,frame_count)

    #Returns smallest & biggest video dimensions
    def image_ranges(self):
        min_w = (10000, 10000)
        max_w = (-1,-1)
        min_h = (10000, 10000)
        max_h = (-1,-1)
        count = 0
        total_w = 0
        total_h = 0
        for root, dirs, files in os.walk(self.directory, topdown=True):
            for file in files:
                #print(file)
                ((width,height),_,_) = self.get_video_specs(os.path.join(root,file),False)
                count += 1
                total_w += width
                total_h += height
                if (width < min_w[0]):
                    min_w = (width, height)

                if (width > max_w[0]):
                    max_w = (width, height)

                if (height < min_h[1]):
                    min_h = (width, height)
                
                if (height > max_h[1]):
                    max_h = (width, height)
        average_size = (total_w/count,total_h/count)
        print("Dimensions range:", min_w , "/", min_h, "--->", max_w , "/", max_h,"and average:", average_size)
        return 


        
    #Return Random Video/Image Directory of selected class
    def get_random_asset(self, target_dir, target_class, isVideo = True):
        
        if isVideo:
            target_dir = self.directory + "/" + target_dir
        else: 
            target_dir = self.frameDirectory + "/" + target_dir
        target_folder = target_dir + "/" + target_class
        random_vid = random.sample(os.listdir(target_folder),1)

        return target_folder + "/" + random_vid[0]

    #Display Random Video fom selected class embedded in Notebook
    def view_random_video(self, target_dir, target_class):
        random_path = self.get_random_asset(target_dir,target_class)
        extension = random_path[-3:]
        if extension == "avi":
            print("popo")
        else:
            Video.from_file(random_path,autoplay = True)
        
        return

    
    #Break Video to Frames and 
    def save_frames(self, vid, path, newfps):
        video = cv2.VideoCapture(vid)

        #calculate at how many intervals you save a picture
        #this could have been more efficient if it was calculated once for RWF dataset
        #but not all datasets have same size videos
        ((_,_),fps,total_frames) = self.get_video_specs(vid, False)
        if fps == 0:
            return
        duration = total_frames/fps
        new_frames = newfps * duration
        step = (int) (total_frames/(new_frames + 1))
        
        #Unfortunately we can't skip frames and that's not only OpenCV's problem
        #Video codes encode based on the previous frame
        i = 0
        success = True
        name, _ = os.path.splitext(os.path.basename(vid))
        while success:
            i += 1
            success, image = video.read()
            if i % step == 0:
                count = i // step 
                filename = name + ("%d.jpg" % count)
                fullpath = os.path.join(path,filename)
                cv2.imwrite(fullpath, image)

            
        return 


    def dataframe(self, fps, name = 'Dataframe'):
        path = self.directory + ' ' + name
        self.frameDirectory = path
        try:
            os.mkdir(path)
            for root, dirs, files in os.walk(self.directory, topdown=True):
                #Replace original folder's name in path with new folder's name
                new_root = root.replace(self.directory,path)
                #Create new directory with same name (as you traverse the file tree topdown)
                for name in dirs:
                    newpath = os.path.join(new_root,name) 
                    os.mkdir(newpath)
                #If we reached the leaves (files) save the specified frames for each file
                if not dirs:
                    newpath = root.replace(self.directory,path) 
                    for file in os.listdir(root):
                        oldpath = os.path.join(root,file)
                        self.save_frames(oldpath,newpath,fps)
                                    
        except:
            print("File %s already exists" % path)
        return


class Visualize:
    #Gets a directory and returns an image as np_array 
    def load(filename):
        np_image = Image.open(filename)
        np_image = np.array(np_image).astype('float32')/255
        np_image = transform.resize(np_image, (224, 224, 3))
        np_image = np.expand_dims(np_image, axis=0)
        return np_image

    #Gets a model & list of image directories and returns models classification prediction for each image
    def predictions(model, imagedirs, classes):
        images = list(map(Visualize.load,imagedirs))
        predictions = np.array(list(map(model.predict, images)))
        classifications = list(map(lambda x: (classes[np.argmax(x)], np.max(x)), predictions))
        return classifications

    #Gets input model's history and plots accuracy/loss curves per epoch
    def plot_loss_curves(histories,titles = None,figsz=(-1,-1)):

        #Returns separate loss curves for training and validation metrics
        for history, title in zip(histories,titles):
            accuracy = history.history['accuracy']
            loss = history.history['loss']
            
            val_accuracy = history.history['val_accuracy']
            val_loss = history.history['val_loss']     
            
            epochs = range(len(history.history['loss']))

            # Plot loss curves
            if figsz==(-1,-1):
                plt.figure(figsize=(14, 7))
            else:
                plt.figure(figsize=figsz)
            if title != None:
                plt.suptitle(title,fontsize=18)
            plt.subplot(1,2,1)
            plt.plot(epochs, loss, label='training_loss')
            plt.plot(epochs, val_loss, label='val_loss')
            plt.title('Loss')
            plt.xlabel('Epochs')
            plt.legend()

            # Plot accuracy curves
            plt.subplot(1,2,2)
            plt.plot(epochs, accuracy, label='training_accuracy')
            plt.plot(epochs, val_accuracy, label='val_accuracy')
            plt.title('Accuracy')
            plt.xlabel('Epochs')
            plt.legend()
            plt.tight_layout()
            plt.show()
        return

    def compare_histories(histories,titles,initial_epochs=5,figsz=(16,5)):
        """
        Plots many histories
        """
        plt.figure(figsize=figsz)
        for history, title in zip(histories,titles):            
            plt.subplot(1, 2, 1)
            acc = history.history["accuracy"]
            plt.plot(acc, label=title)

        plt.legend(loc='lower right')
        plt.title('Accuracy')
        plt.xlabel('epoch')

        for history, title in zip(histories,titles):            
            plt.subplot(1, 2, 2)
            loss = history.history["loss"]
            plt.plot(loss, label=title)

        plt.legend(loc='lower right')
        plt.title('Loss')
        plt.xlabel('epoch')

        plt.show()


        plt.figure(figsize=figsz)
        for history, title in zip(histories,titles):            
            plt.subplot(1, 2, 1)
            val_acc = history.history["val_accuracy"]
            plt.plot(val_acc, label=title)

        plt.legend(loc='lower right')
        plt.title('Validation Accuracy')
        plt.xlabel('epoch')

        for history, title in zip(histories,titles):            
            plt.subplot(1, 2, 2)
            val_loss = history.history["val_loss"]
            plt.plot(val_loss, label=title)

        plt.legend(loc='lower right')
        plt.title('Validation Loss')
        plt.xlabel('epoch')

        plt.show()

        return
        

class Callbacks:
    # Create tensorboard callback   
    def create_tensorboard_callback(dir_name, experiment_name):
        log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        )
        print(f"Saving TensorBoard log files to: {log_dir}")
        return tensorboard_callback

    #Save checkpoints/Weights only 
    def checkpoint_callback(checkpoint_path = "Checkpoints",weights = True,freq = "epoch"):
        try:
            full_path = checkpoint_path + '/' + 'checkpoint.ckpt'
            os.mkdir(checkpoint_path)
        finally:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=full_path,
                                                         save_weights_only=weights, # set to False to save the entire model
                                                         save_best_only=False, # set to True to save only the best model instead of a model every epoch 
                                                         save_freq=freq, # save every epoch
                                                         verbose=1)
        return checkpoint
