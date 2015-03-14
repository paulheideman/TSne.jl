include("../src/TSne.jl")
using RDatasets
using Gadfly
using TSne

lables = ()

println("Using Iris dataset.")
iris = dataset("datasets", "iris")
X = array(iris[:,1:4])
labels = iris[:,5]
plotname = "iris"
initial_dims = -1
iterations = 1500
perplexity = 15

println("X dimensions are: " * string(size(X)))
Y = tsne(X, 2, initial_dims, iterations, perplexity)
println("Y dimensions are: " * string(size(Y)))

theplot = plot(x=Y[:,1], y=Y[:,2], color=labels)

draw(PDF(plotname*".pdf", 4inch, 3inch), theplot)
draw(SVG(plotname*".svg", 4inch, 3inch), theplot)
