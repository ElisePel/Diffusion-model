# create our unet model
unet = Unet(channels=1)

# initialize the model in the memory of our GPU
test_images = X[0].reshape(1, 32, 32, 1)
test_timestamps = generate_timestamp(0, 1)
k = unet(test_images, test_timestamps)

# create our optimizer, we will use adam with a Learning rate of 1e-4
opt = keras.optimizers.Adam(learning_rate=1e-5)

def loss_fn(real, generated):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(real, generated)
    return loss
  
rng = 0

def train_step(batch):
    rng, tsrng = np.random.randint(0, 1000, size=(2,)) #100000
    timestep_values = generate_timestamp(tsrng, batch.shape[0])

    noised_image, noise = forward_noise(rng, batch, timestep_values)
    with tf.GradientTape() as tape:
        prediction = unet(noised_image, timestep_values)
        
        loss_value = loss_fn(noise, prediction)
    
    gradients = tape.gradient(loss_value, unet.trainable_variables)
    opt.apply_gradients(zip(gradients, unet.trainable_variables))

    return loss_value
  
t1 = time.time()
epochs = 30
list_avg = []
for e in range(1, epochs+1):
    # this is cool utility in Tensorflow that will create a nice looking progress bar
    bar = tf.keras.utils.Progbar(120-1)
    losses = []
    for i, batch in enumerate(iter(X_train)):
        # run the training loop
        loss = train_step(batch)
        losses.append(loss)
        bar.update(i, values=[("loss", loss)])

    avg = np.mean(losses)
    list_avg.append(avg)
    print(f"Average loss for epoch {e}/{epochs}: {avg}")
t2 = time.time()
