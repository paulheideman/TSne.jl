module TSne

# Numpy Math.sum => axis = 0 => sum down the columns. axis = 1 => sum along the rows
# Julia Base.sum => axis = 1 => sum down the columns. axis = 2 => sum along the rows

#
#  tsne.jl
#
#  This is a straight off Julia port of Laurens van der Maatens python implementation of tsne

export tsne, pca

using Distances

function Hbeta(D, beta = 1.0)
	#Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution.
	P = exp(-copy(D) * beta)
	sumP = sum(P)
	H = log(sumP) + beta * sum(D .* P) / sumP
	P = P / sumP
	return (H, P)
end

function x2p(X, tol = 1e-5, perplexity = 30.0)
	#Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity.
	println("Computing pairwise distances...")
	(n, d) = size(X)
	sum_X = sum((X.^2),2)
	D = (-2 * (X * X') .+ sum_X)' .+ sum_X
	P = zeros(n, n)
	beta = ones(n, 1)
	logU = log(perplexity)

	# Loop over all datapoints
	range = [1:n]
	for i in 1:n

		# Print progress
		if mod(i, 500) == 0
			println("Computing P-values for point " * string(i) *  " of " * string(n) * "...")
        	end

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -Inf
		betamax =  Inf

		inds = range[range .!=i]
		Di = D[i, inds]
		(H, thisP) = Hbeta(Di, beta[i])

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU
		tries = 0
		while abs(Hdiff) > tol && tries < 50

			# If not, increase or decrease precision
			if Hdiff > 0
				betamin = beta[i]
				if betamax == Inf || betamax == -Inf
					beta[i] = beta[i] * 2
				else
					beta[i] = (beta[i] + betamax) / 2
				end
			else
				betamax = beta[i]
				if betamin == Inf || betamin == -Inf
					beta[i] = beta[i] / 2
				else
					beta[i] = (beta[i] + betamin) / 2
				end
			end

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i])
			Hdiff = H - logU
			tries = tries + 1
		end
		# Set the final row of P
  		P[i, inds] = thisP

	end
	# Return final P-matrix
	println("Mean value of sigma: " * string(mean(sqrt(1 ./ beta))))
	return P
end

function pca(X, no_dims = 50)
	#Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions.

	println("Preprocessing the data using PCA...")
	(n, d) = size(X)
	X = X - repmat(mean(X, 1), n, 1)
	C = (X' * X) ./ (size(X,1)-1)
	(l, M) = eig(C)
	sorder = sortperm(l,rev=true)
	M = M[:,sorder]
	ret_dims = no_dims > d ? d : no_dims
	Y = X * M[:,1:ret_dims]
	return Y
end

function grad!(dY, Y, P, n, no_dims, Q)
  fill!(dY, 0.0)
  pairwise!(Q, SqEuclidean(), Y')
  sum_Q = 0.0
  for col = 1:n, row = 1:n
    if col != row
      Q[row, col] = 1 / (1 + Q[row, col])
      sum_Q += Q[row, col]
    end
  end
  for col = 1:n, row = 1:n
    mult = (P[row, col] - (Q[row, col] / sum_Q)) * Q[row, col]
    for d = 1:no_dims
      dY[col, d] += (Y[col, d] - Y[row, d]) * mult
    end
  end
end

function cost!(P, Q, n)
  # destroys Q
  sum_Q = sum(Q)
  removeNA = (x) -> isnan(x) ? 0.0 : x
  for col = 1:n, row = 1:n
    v = P[row, col] * log(P[row, col] / max(Q[row, col] /  sum_Q, 1e-12))
    Q[row, col] = removeNA(v)
  end
  sum(Q)
end

function update!(Y, dY, iY, gains, momentum, min_gain, eta, n, no_dims)
  mean_Y = zeros(no_dims)
  for d = 1:no_dims, row = 1:n
    gains[row, d] = (gains[row, d] + 0.2) * ((dY[row, d] .> 0) != (iY[row, d] > 0)) +
                    (gains[row, d] * 0.8) * ((dY[row, d] .> 0) == (iY[row, d] > 0))
    gains[row, d] = gains[row, d] < min_gain ? min_gain : gains[row, d]
    iY[row, d] = momentum * iY[row, d] - eta * (gains[row, d] * dY[row, d])
    Y[row, d] = Y[row, d] + iY[row, d]
    mean_Y[d] += Y[row, d]
  end
  for d = 1:no_dims
    mean_Y[d] /= n
  end
  for d= 1:no_dims, row = 1:n
    Y[row, d] = Y[row, d] - mean_Y[d]
  end
end

function tsne(X, no_dims = 2, initial_dims = -1, max_iter = 1000, perplexity = 30.0)
	#Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	# Diffrent from orginal, default is to not use PCA
	println("Initial X Shape is : " * string(size(X)))

	# Initialize variables
	if(initial_dims>0)
		X = pca(X, initial_dims)
	end
        # zero the mean
        X = X .- mean(X)
        # scale
        X = X / std(X)
	(n, d) = size(X)
	initial_momentum = 0.5
	final_momentum = 0.8
	eta = 500
	min_gain = 0.01
	Y = randn(n, no_dims) .* 0.0001
	dY = zeros(n, no_dims)
	iY = zeros(n, no_dims)
	gains = ones(n, no_dims)
        Q = zeros(n, n)

	# Compute P-values
	P = x2p(X, 1e-5, perplexity)
	P = P + P'
	P = P / sum(P)
	P = P * 12						# early exaggeration
	P = max(P, 1e-12)
	
	# Run iterations
	for iter in 1:max_iter
                grad!(dY, Y, P, n, no_dims, Q)

		# Perform the update
		if iter <= 250
			momentum = initial_momentum
		else
			momentum = final_momentum
		end
                update!(Y, dY, iY, gains, momentum, min_gain, eta, n, no_dims)

		# Compute current value of cost function
		if mod((iter + 1), 10) == 0
			println("Iteration ", (iter + 1), ": error is ", cost!(P, Q, n))
		end
		# Stop lying about P-values
		if iter == 250
			P = P / 12
		end
    	end
	# Return solution
	return Y
	end

end
