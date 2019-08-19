using Flux, Flux.Data.MNIST, Statistics
using Flux: Tracker, throttle, params, binarycrossentropy, crossentropy
using Flux.Tracker: update!
using NNlib: relu, leakyrelu
using Base.Iterators: partition
using Images: channelview

include("data_loader.jl")

BATCH_SIZE = 512
REAL_LABEL =  ones(1, BATCH_SIZE)
FAKE_LABEL = zeros(1, BATCH_SIZE)

train_data = load_dataset_as_batches("../celeba-dataset/img_align_celeba/img_align_celeba/", BATCH_SIZE)
train_data = gpu.(train_data)
println(size(train_data))

NUM_EPOCHS = 50
training_steps = 0

disc_encoder = Chain(Conv((4, 4), 3=>64, stride = (2, 2), pad = (1, 1)), 
					x -> leakyrelu.(x, 0.2),
					Conv((4, 4), 64=>128, stride = (2, 2), pad = (1, 1)),
					BatchNorm(128),
					x -> leakyrelu.(x, 0.2),
					Conv((4, 4), 128=>256, stride = (2, 2), pad = (1, 1)),
					BatchNorm(256),
					x -> leakyrelu.(x, 0.2))

disc_decoder = Chain(ConvTranspose((4, 4), 256=>128, stride = (2, 2), pad = (1, 1)),
					BatchNorm(128),
					x -> leakyrelu.(x, 0.2),
					ConvTranspose((4, 4), 128=>64, stride = (2, 2), pad = (1, 1)),
					BatchNorm(64),
					x ->leakyrelu.(x, 0.2),
					ConvTranspose((4, 4), 64=>3, stride = (2, 2), pad = (1, 1)),
					BatchNorm(3),
					x -> tanh.(x))

discriminator = Chain(disc_encoder, disc_decoder)

generator = Chain(ConvTranspose((1, 1), 100=>1024*16),
				x -> reshape(x, 4, 4, 1024, :),
				ConvTranspose((4, 4), 1024=>512, stride = (2, 2), pad = (1, 1)),
				BatchNorm(512),
				x -> relu.(x),
				ConvTranspose((4, 4), 512=>256, stride = (2, 2), pad = (1, 1)),
				x -> relu.(x),
				ConvTranspose((4, 4), 256=>128, stride = (2, 2), pad = (1, 1)),
				x -> relu.(x),
				ConvTranspose((4, 4), 128=>3, stride = (2, 2), pad = (1, 1)),
				x -> tanh.(x))


m = 20.0f0
function disc_loss(X)
	output = discriminator(X)
	sample_z = randn(1, 1, 100, BATCH_SIZE)
	x_fake = generator(sample_z)
	disc_loss_real = Flux.mse(output, X)
	disc_loss_fake = Flux.mse(discriminator(x_fake), x_fake)
	disc_loss_fake += relu.(m - disc_loss_fake)
	return disc_loss_real + disc_loss_fake
end 

function generator_loss(X)
	sample_z = randn(1, 1, 100, BATCH_SIZE)
	x_fake = generator(sample_z)
	return Flux.mse(x_fake, discriminator(x_fake))
end

opt_disc = ADAM()
opt_gen = ADAM()

function training(X)
	disc_grad = Flux.Tracker.gradient(()->disc_loss(X), params(discriminator))
	gen_grad = Flux.Tracker.gradient(()->generator_loss(X), params(generator))

	update!(opt_disc, params(discriminator), disc_grad)
	update!(opt_gen, params(generator), gen_grad)

	return disc_loss(X), generator_loss(X)
end

NUM_EPOCHS = 50

function train()
  i = 0
  for epoch in 1:NUM_EPOCHS
    println("---------------EPOCH : $epoch----------------")
    for d in train_data
      disc_loss, generator_loss = training(d |> gpu)
      println("Discriminator loss : $disc_loss, Generator loss : $generator_loss")
      i += 1
      if i % 500 == 0
        save_weights(discriminator_logit, discriminator_classifier, generator)
      end
    end
  end
end





