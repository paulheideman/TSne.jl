include("../src/TSne.jl")
using MNIST
using Gadfly
using TSne

lables = ()

println("Using MNIST dataset.")
X, labels = traindata()
random_idxs = map(int, shuffle(linspace(1, size(X)[2], size(X)[2])))[1:6000]
X = X[:, random_idxs]'
labels = labels[random_idxs]
plotname = "mnist"
initial_dims = 50
iterations = 10000
perplexity = 30

println("X dimensions are: " * string(size(X)))
Y = tsne(X, 2, initial_dims, iterations, perplexity)
println("Y dimensions are: " * string(size(Y)))

theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)

draw(PDF(plotname*".pdf", 4inch, 3inch), theplot)
draw(SVG(plotname*".svg", 4inch, 3inch), theplot)
