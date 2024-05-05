import numpy as np
import cv2
import os
from keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    LeakyReLU, Conv2DTranspose, Reshape,
    BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam


class DRPredictor:
    directory = 'dataset'
    image_data_file = 'img_data.npy'
    image_label_file = 'img_label.npy'
    cnn_weights_path = 'model/train.weights.h5'
    cnn_json_model_path = "model/train.json"
    gan_model_path = 'model/generator_model_{}.keras'
    latent_dim = 200
    n_epoch = 100
    n_batch = 128

    def prepare_dataset(self):
        images = []
        image_labels = []
        for label in os.listdir(self.directory):
            if label == '.DS_Store':
                continue
            for file in os.listdir(os.path.join(self.directory, label)):
                if label == '.DS_Store':
                    continue
                img = np.array(
                    cv2.resize(
                        cv2.imread(
                            os.path.join(
                                os.path.join(
                                    self.directory, label
                                ), file
                            )
                        ), (32, 32)
                    )
                ).reshape(32, 32, 3)
                images.append(img)
                image_labels.append(int(label))

        X = np.asarray(images).astype('float32')/255
        Y = utils.to_categorical(np.asarray(image_labels))
        np.save(self.image_data_file, X)
        np.save(self.image_label_file, Y)

    def create_cnn_classifier_model(self):
        # model = Sequential([
        #     Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        #     BatchNormalization(),
        #     MaxPooling2D((2, 2)),
        #     Conv2D(64, (3, 3), activation='relu'),
        #     BatchNormalization(),
        #     MaxPooling2D((2, 2)),
        #     Conv2D(128, (3, 3), activation='relu'),
        #     BatchNormalization(),
        #     MaxPooling2D((2, 2)),
        #     Flatten(),
        #     Dense(128, activation='relu'),
        #     Dropout(0.5),
        #     Dense(3, activation='softmax')
        # ])

        model = Sequential([
            Input(shape=(32, 32, 3)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_image_data(self):
        return (np.load(self.image_data_file), np.load(self.image_label_file))

    def train_cnn_model(self):
        self.prepare_dataset()
        x, y = self.load_image_data()
        classifier = self.create_cnn_classifier_model()
        classifier.fit(x, y, batch_size=32, epochs=self.n_epoch)
        classifier.save_weights(self.cnn_weights_path)
        model_json = classifier.to_json()
        with open(self.cnn_json_model_path, "w") as json_file:
            json_file.write(model_json)

    def create_discriminator_model(self, in_shape=(32, 32, 3)):
        model = Sequential([
            Input(shape=in_shape),
            Conv2D(64, (3, 3), padding='same'),
            LeakyReLU(negative_slope=0.2),
            Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(negative_slope=0.2),
            Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(negative_slope=0.2),
            Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(negative_slope=0.2),
            Flatten(),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        return model

    def create_generator_model(self):
        n_nodes = 256 * 4 * 4
        model = Sequential([
            Input(shape=(self.latent_dim,)),
            Dense(n_nodes),
            LeakyReLU(negative_slope=0.2),
            Reshape((4, 4, 256)),
            Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.2),
            Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.2),
            Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.2),
            Conv2DTranspose(3, (3, 3), activation='tanh', padding='same')
        ])
        return model
    
    # def create_discriminator_model(self, in_shape=(32, 32, 3)):
    #     model = Sequential([
    #         Conv2D(64, (3, 3), padding='same', input_shape=in_shape),
    #         LeakyReLU(negative_slope=0.2),
    #         Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    #         LeakyReLU(negative_slope=0.2),
    #         Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    #         LeakyReLU(negative_slope=0.2),
    #         Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
    #         LeakyReLU(negative_slope=0.2),
    #         Flatten(),
    #         Dropout(0.4),
    #         Dense(1, activation='sigmoid')
    #     ])
    #     model.compile(
    #         loss='binary_crossentropy',
    #         optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    #         metrics=['accuracy']
    #     )
    #     return model

    # def create_generator_model(self):
    #     n_nodes = 256 * 4 * 4
    #     model = Sequential([
    #         Dense(n_nodes, input_dim=self.latent_dim),
    #         LeakyReLU(negative_slope=0.2),
    #         Reshape((4, 4, 256)),
    #         Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    #         LeakyReLU(negative_slope=0.2),
    #         Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    #         LeakyReLU(negative_slope=0.2),
    #         Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    #         LeakyReLU(negative_slope=0.2),
    #         Conv2DTranspose(3, (3, 3), activation='tanh', padding='same')
    #     ])
    #     return model

    def create_gan_model(self, g_model, d_model):
        d_model.trainable = False
        model = Sequential([g_model, d_model])
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
        )
        return model

    def load_image_data_real(self):
        return (
            (np.load(self.image_data_file).astype('float32') - 127.5) / 127.5
        )

    def generate_real_samples(self, dataset, half_batch):
        return (
            dataset[np.random.randint(0, dataset.shape[0], half_batch)],
            np.ones((half_batch, 1))
        )

    def generate_fake_samples(self, g_model, n_samples):
        return (
            g_model.predict(
                self.generate_latent_points(n_samples)
            ),
            np.zeros((n_samples, 1))
        )

    def generate_latent_points(self, n_samples):
        return (
            np.random
            .randn(self.latent_dim, n_samples)
            .reshape(n_samples, self.latent_dim)
        )

    def generate_gan_samples(self, n_samples):
        return (
            self.generate_latent_points(n_samples),
            np.ones((n_samples, 1))
        )

    def train_gan_model(self):
        dataset = self.load_image_data_real()
        d_model = self.create_discriminator_model()
        g_model = self.create_generator_model()
        gan_model = self.create_gan_model(g_model, d_model)
        batch_size = dataset.shape[0] // self.n_batch
        half_batch = self.n_batch // 2
        for epoch in range(1, self.n_epoch+1):
            for batch in range(1, batch_size+1):
                x_real, y_real = self.generate_real_samples(
                    dataset, half_batch
                )
                x_fake, y_fake = self.generate_fake_samples(
                    g_model, half_batch
                )
                x_gan, y_gan = self.generate_gan_samples(
                    self.n_batch
                )
                d_loss_real, _ = d_model.train_on_batch(x_real, y_real)
                d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)
                g_loss = gan_model.train_on_batch(x_gan, y_gan)
                print(
                    f'Epoch:{epoch}, {batch}/{batch_size}, '
                    f'd_loss_real = {d_loss_real}, '
                    f'd_loss_fake = {d_loss_fake}, '
                    f'g_loss = {g_loss}'
                )
            if epoch % 10 == 0:
                x_real, y_real = self.generate_real_samples(
                    dataset, half_batch
                )
                x_fake, y_fake = self.generate_fake_samples(
                    g_model, half_batch
                )
                _, acc_real = d_model.evaluate(x_real, y_real)
                _, acc_fake = d_model.evaluate(x_fake, y_fake)
                print(
                    f'>Accuracy real: {acc_real*100}%, fake: {acc_fake*100}%'
                )
                g_model.save(self.gan_model_path.format(str(epoch).zfill(3)))


if __name__ == '__main__':
    dr_obj = DRPredictor()
    dr_obj.train_cnn_model()
    dr_obj.train_gan_model()
