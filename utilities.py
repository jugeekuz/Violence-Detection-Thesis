import cv2
import os
import random
from ipywidgets import Video
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import datetime
import tensorflow as tf

class Preprocess:
    def __init__(self, directory, frameDirectory ="", isVideo=True):
        self.directory = directory
        self.frameDirectory = frameDirectory
        self.isVideo = isVideo

    def plot_album(self, figsz, image_paths, nrows, ncols, titles, resz = (-1,-1)):
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
            fig.tight_layout()        
        return


    #Return Selected Videos : Dimensions, Duration, Framerate and print it
    def get_video_specs(self, vid, toPrint = True):
        if(not self.isVideo):
            print("This directory doesn't contain videos")
            return

        video = cv2.VideoCapture(vid)

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        width = video.get(3)
        height = video.get(4)
        if toPrint:
            print(f"The video has size of ({width},{height}) , duration of: {frame_count/fps} sec FPS: {fps} and total frames: {frame_count}.")
        
        return ((width,height),fps,frame_count)

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

    #def view_consecutive_frames(self, target_dir, target_class, ):
    
    #Break Video to Frames and 
    def save_frames(self, vid, path, newfps):
        video = cv2.VideoCapture(vid)

        ((_,_),fps,total_frames) = self.get_video_specs(vid, False)
        step = (int) (fps/newfps)
        success, image = video.read()
        count = 0
        notskip = 0
        while success:
            name, _ = os.path.splitext(os.path.basename(vid))
            name += "%d.jpg" % count
            fullpath = os.path.join(path,name)

            notskip += 1
            if notskip == step:
                cv2.imwrite(fullpath, image)
                notskip = 0
            success, image = video.read()
            #print('Read a new frame: ',success)

            count += 1
        return 


    def dataframe(self, folder, fps):
        newname = ' Dataframe'
        path = folder + newname
        self.frameDirectory = path
        try:
            os.mkdir(path)
            for root, dirs, files in os.walk(folder, topdown=True):
                new_root = root.replace(folder,path)
                for name in dirs:
                    newpath = os.path.join(new_root,name) 
                    os.mkdir(newpath)
                if not dirs:
                    newpath = root.replace(folder,path) 
                    for file in os.listdir(root):
                        oldpath = os.path.join(root,file)
                        self.save_frames(oldpath,newpath,fps)
                                    
        except:
            print("File %s already exists" % path)
        return

class Visualize:

    #Gets input model's history and plots accuracy/loss curves per epoch
    def plot_loss_curves(histories):

        #Returns separate loss curves for training and validation metrics.
        for history in histories:
            accuracy = history.history['accuracy']
            loss = history.history['loss']
            
            val_accuracy = history.history['val_accuracy']
            val_loss = history.history['val_loss']     
            
            epochs = range(len(history.history['loss']))

            # Plot loss curves
            plt.plot(epochs, loss, label='training_loss')
            plt.plot(epochs, val_loss, label='val_loss')
            plt.title('Loss')
            plt.xlabel('Epochs')
            plt.legend()

            # Plot accuracy curves
            plt.figure()
            plt.plot(epochs, accuracy, label='training_accuracy')
            plt.plot(epochs, val_accuracy, label='val_accuracy')
            plt.title('Accuracy')
            plt.xlabel('Epochs')
            plt.legend();
        return

    def compare_historys_finetuning(original_history, new_history, initial_epochs=5):
        """
        Compares two model history objects.
        """
        # Get original history measurements
        acc = original_history.history["accuracy"]
        loss = original_history.history["loss"]

        print(len(acc))

        val_acc = original_history.history["val_accuracy"]
        val_loss = original_history.history["val_loss"]

        # Combine original history with new history
        total_acc = acc + new_history.history["accuracy"]
        total_loss = loss + new_history.history["loss"]

        total_val_acc = val_acc + new_history.history["val_accuracy"]
        total_val_loss = val_loss + new_history.history["val_loss"]

        print(len(total_acc))
        print(total_acc)

        # Make plots
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(total_acc, label='Training Accuracy')
        plt.plot(total_val_acc, label='Validation Accuracy')
        plt.plot([initial_epochs-1, initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(total_loss, label='Training Loss')
        plt.plot(total_val_loss, label='Validation Loss')
        plt.plot([initial_epochs-1, initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
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
