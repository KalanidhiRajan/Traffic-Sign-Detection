from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image

import pickle
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
data_dir = 'C:/Users/testi/Desktop/mp/try/Indian/Traffic Sign Dataset/train'
# Using ImageDataGenerator to load the Images for Training and Testing the CNN Model
datagenerator = {
    "train": ImageDataGenerator(horizontal_flip=True,
                                vertical_flip=True,
                                rescale=1. / 255,
                                validation_split=0.1,
                                shear_range=0.1,
                                zoom_range=0.1,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rotation_range=30,
                               ).flow_from_directory(directory=data_dir,
                                                     target_size=(300, 300),
                                                     subset='training',
                                                    ),

    "valid": ImageDataGenerator(rescale=1 / 255,
                                validation_split=0.1,
                               ).flow_from_directory(directory=data_dir,
                                                     target_size=(300, 300),
                                                     subset='validation',
                                                    ),
}
base_model = InceptionV3(weights=None, include_top=False, input_shape=(300, 300, 3))

# Load Weights for the InceptionV3 Model
base_model.load_weights('C:/Users/testi/Downloads/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Setting the Training of all layers of InceptionV3 model to false
base_model.trainable = False
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.15),
    Dense(1024, activation='relu'),
    Dense(85, activation='softmax') # 85 Output Neurons for 85 Classes
])
opt = optimizers.Adam(learning_rate=0.0001)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
model.summary()


from keras.utils.vis_utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True)
batch_size = 32
epochs = 10

train_generator = datagenerator["train"]
valid_generator = datagenerator["valid"]

steps_per_epoch = train_generator.n // batch_size
validation_steps = valid_generator.n // batch_size

print("steps_per_epoch :", steps_per_epoch)
print("validation_steps :", validation_steps)
steps_per_epoch : 125
validation_steps : 12
filepath = "./model_{epoch:02d}-{val_accuracy:.2f}.h5"

checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint1]

history = model.fit_generator(generator=train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                              validation_data=valid_generator, validation_steps=validation_steps,
                              callbacks=callbacks_list)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# ________________ Graph 1 -------------------------

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

# ________________ Graph 2 -------------------------

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()
test_loss, test_acc = model.evaluate(valid_generator)
print('test accuracy : ', test_acc)
# Check our folder and import the model with best validation accuracy
loaded_best_model = keras.models.load_model("./model_09-0.53.h5")

# Custom function to load and predict label for the image
def predict(img_rel_path):
    # Import Image from the path with size of (300, 300)
    img = image.load_img(img_rel_path, target_size=(300, 300))

    # Convert Image to a numpy array
    img = image.img_to_array(img, dtype=np.uint8)

    # Scaling the Image Array values between 0 and 1
    img = np.array(img)/255.0

    # Plotting the Loaded Image
    plt.title("Loaded Image")
    plt.axis('off')
    plt.imshow(img.squeeze())
    plt.show()

    # Get the Predicted Label for the loaded Image
    p = loaded_best_model.predict(img[np.newaxis, ...])

     # Label array
    labels = {0: 'ALL_MOTOR_VEHICLE_PROHIBITED', 1: 'AXLE_LOAD_LIMIT', 2: 'BARRIER_AHEAD', 3: 'BULLOCK_AND_HANDCART_PROHIBITED', 4: 'Dragonfly', 5: 'CATTLE',
              6: 'COMPULSARY_AHEAD', 7: 'COMPULSARY_AHEAD_OR_TURN_LEFT', 8: 'COMPULSARY_AHEAD_OR_TURN_RIGHT', 9: 'COMPULSARY_CYCLE_TRACK',10: 'COMPULSARY_KEEP_RIGHT',11: 'COMPULSARY_KEEP_RIGHT',12: 'COMPULSARY_MINIMUM_SPEED',13: 'COMPULSARY_SOUND_HORN',14: 'COMPULSARY_TURN_LEFT',
             15: 'COMPULSARY_TURN_LEFT_AHEAD',16: 'COMPULSARY_TURN_RIGHT',17: 'COMPULSARY_TURN_RIGHT_AHEAD',18: 'CROSS_ROAD',19: 'CYCLE_CROSSING',20: 'CYCLE_PROHIBITED',21: 'DANGEROUS_DIP',
             22: 'DIRECTION',23: 'FALLING_ROCKS',24: 'FERRY',25: 'GAP_IN_MEDIAN',26: 'GIVE_WAY',27: 'GUARDED_LEVEL_CROSSING',28: 'HANDCART_PROHIBITED',
             29: 'HEIGHT_LIMIT',30: 'HORN_PROHIBITED',31: 'HUMP_OR_ROUGH_ROAD',32: 'LEFT_HAIR_PIN_BEND',33: 'LEFT_HAND_CURVE',34: 'LEFT_REVERSE_BEND',35: 'LEFT_TURN_PROHIBITED',
             36: 'LENGTH_LIMIT',37: 'LOAD_LIMIT',38: 'LOOSE_GRAVEL',39: 'MEN_AT_WORK',40: 'NARROW_BRIDGE',41: 'NARROW_ROAD_AHEAD',42: 'NO_ENTRY',
             43: 'NO_PARKING',44: 'NO_STOPPING_OR_STANDING',45: 'OVERTAKING_PROHIBITED',46: 'PASS_EITHER_SIDE',47: 'PEDESTRIAN_CROSSING',48: 'PEDESTRIAN_PROHIBITED',49: 'PRIORITY_FOR_ONCOMING_VEHICLES',
             50: 'QUAY_SIDE_OR_RIVER_BANK',51: 'RESTRICTION_ENDS',52: 'RIGHT_HAIR_PIN_BEND',53: 'RIGHT_HAND_CURVE',54: 'RIGHT_REVERSE_BEND',55: 'RIGHT_TURN_PROHIBITED',56: 'ROAD_WIDENS_AHEAD',
             57: 'ROUNDABOUT',58: 'SCHOOL_AHEAD',59: 'SIDE_ROAD_LEFT',60: 'SIDE_ROAD_RIGHT',61: 'SLIPPERY_ROAD',62: 'SPEED_LIMIT_15',63: 'SPEED_LIMIT_20',
             64: 'SPEED_LIMIT_30',65: 'SPEED_LIMIT_40',66: 'SPEED_LIMIT_5',67: 'SPEED_LIMIT_50',68: 'SPEED_LIMIT_60',69: 'SPEED_LIMIT_70',70: 'SPEED_LIMIT_80',
             71: 'STAGGERED_INTERSECTION',72: 'STEEP_ASCENT',73: 'STEEP_DESCENT',74: 'STOP',75: 'STRAIGHT_PROHIBITED',76: 'TONGA_PROHIBITED',77: 'TRAFFIC_SIGNAL',
             78: 'TRUCK_PROHIBITED',79: 'TURN_RIGHT',80: 'T_INTERSECTION',81: 'UNGUARDED_LEVEL_CROSSING',82: 'U_TURN_PROHIBITED',83: 'WIDTH_LIMIT',84: 'Y_INTERSECTION',}
    
    print("\n\nMaximum Probability: ", np.max(p[0], axis=-1))
    predicted_class = labels[np.argmax(p[0], axis=-1)]
    print("Classified:", predicted_class, "\n\n")

    classes=[]
    prob=[]
    print("\n-------------------Individual Probability--------------------------------\n")

    for i,j in enumerate (p[0],0):
        print(labels[i].upper(),':',round(j*100,2),'%')
        classes.append(labels[i])
        prob.append(round(j*100,2))
        
    def plot_bar_x():
        # this is for plotting purpose
        index = np.arange(len(classes))
        plt.bar(index, prob)
        plt.xlabel('Labels', fontsize=8)
        plt.ylabel('Probability', fontsize=8)
        plt.xticks(index, classes, fontsize=8, rotation=20)
        plt.title('Probability for loaded image')
        plt.show()
    plot_bar_x()

predict("C:/Users/testi/Desktop/mp/try/Indian/Traffic Sign Dataset/test/TURN_RIGHT/80040.jpg")
