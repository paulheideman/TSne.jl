include("../src/TSne.jl")
if Pkg.installed("MNIST") == nothing
  Pkg.clone("https://github.com/johnmyleswhite/MNIST.jl")
end
using RDatasets
using Gadfly
using TSne
using MNIST

use_iris = false
lables = ()

if use_iris
	println("Using Iris dataset.")
	iris = dataset("datasets", "iris")
	X = array(iris[:,1:4])
	labels = iris[:,5]
	plotname = "iris"
	initial_dims = -1
	iterations = 1500
	perplexity = 15
else
	println("Using MNIST dataset.")
	X, labels = traindata()
        random_idxs = map(int, shuffle(linspace(1, size(X)[2], size(X)[2])))[1:6000]
        X = X[:, random_idxs]
        labels = labels[random_idxs]
	plotname = "mnist"
	initial_dims = 50
	iterations = 10000
	perplexity = 30
end


#X = randn(5, 3)

println("X dimensions are: " * string(size(X)))
Y = tsne(X, 2, initial_dims, iterations, perplexity)
#Y = pca(X,2)
println("Y dimensions are: " * string(size(Y)))

theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)

draw(PDF(plotname*".pdf", 4inch, 3inch), theplot)
draw(SVG(plotname*".svg", 4inch, 3inch), theplot)
