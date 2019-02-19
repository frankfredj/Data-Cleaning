Fix_data <- function(train.path, test.path, numeric.max.missing, unique.vals, scale.numerics, id.col){

#Load dependencies

require(doParallel)
require(readr)
require(missForest)
require(caret)
require(forecast)
require(e1071)

ptm <- proc.time()

#Load data

print("uploading data...", quote = FALSE)

train <- read.csv(train.path)
test <- read.csv(test.path)

#Remove id column

if(id.col){

		train <- train[,-1]
		test <- test[,-1]

}

#Retrieve the target variable

target.var <- train[,ncol(train)]
train <- train[,-ncol(train)]

#Match predictors

print("processing data...", quote = FALSE)

match.index <- match(colnames(train), colnames(test))
		test <- test[, match.index]
		match.index <- NULL

#Separate factors from numeric data

types <- matrix(nrow = 1, ncol = ncol(train))
		for(i in 1:ncol(train)){types[i] <- class(train[,i])}
		
numerics.index <- which(types == "integer" | types == "numeric")
factors.index <- c(1:ncol(train))[-numerics.index]

#Convert low-unique numerics to factors

temp.frame <- train[,numerics.index]
		unique.perc <- function(x){return(length(unique(x)))}

		temp.vars <- apply(temp.frame, 2, unique.perc)
		temp.vars <- which(temp.vars <= unique.vals)

if(length(temp.vars) != 0){
		factors.index <- c(factors.index, numerics.index[temp.vars])
		numerics.index <- numerics.index[-temp.vars]
}

temp.frame <- NULL
temp.vars <- NULL

#Drop numeric variables with high na %

temp.frame.train <- train[,numerics.index]
temp.frame.test <- test[,numerics.index]

miss.p <- function(x){return(length(which(is.na(x))) / length(x))}
		miss.perc.train <- apply(temp.frame.train, 2, miss.p)
		miss.perc.test <- apply(temp.frame.test, 2, miss.p)

rmv <- unique(c(which(miss.perc.train >= numeric.max.missing), which(miss.perc.test >= numeric.max.missing)))

if(length(rmv) != 0){numerics.index <- numerics.index[-rmv]}


#One-hot encode factor variables

one.hot.encode <- function(reference, target, index){

ref.vals <- unique(reference)
tar.vals <- unique(target)

if(class(ref.vals) == "factor"){ref.vals <- levels(ref.vals)}
if(class(tar.vals) == "factor"){tar.vals <- levels(tar.vals)}

keep <- ref.vals[which(!is.na(match(ref.vals, tar.vals)))]
if(class(keep) == "factor"){keep <- levels(keep)}

m <- length(keep)

if(m > 1){

ref.out <- matrix(nrow = length(reference), ncol = m)
tar.out <- matrix(nrow = length(target), ncol = m)

for(i in 1:m){

ref.match <- which(reference == keep[i])
tar.match  <- which(target == keep[i])

ref.out[ref.match,i] <- 1
ref.out[-ref.match,i] <- 0

tar.out[tar.match,i] <- 1
tar.out[-tar.match,i] <- 0

}

col.names <- paste(colnames(train)[index], c(1:m), sep = "")
		colnames(ref.out) <- col.names
		colnames(tar.out) <- col.names

return(list(ref = ref.out, tar = tar.out))} else {return(list(ref = NULL, tar = NULL))}

}


cl <- makeCluster(detectCores()-2)
registerDoParallel(cl)

factors.OHE <- foreach(i = 1:length(factors.index)) %dopar% one.hot.encode(train[,factors.index[i]], test[,factors.index[i]], factors.index[i])

stopCluster(cl)


train.OHE <- matrix(nrow = nrow(train), ncol = 1)
test.OHE <- matrix(nrow = nrow(test), ncol = 1)

for(i in 1:length(factors.OHE)){

		train.OHE <- cbind(train.OHE, factors.OHE[[i]]$ref)
		test.OHE <- cbind(test.OHE, factors.OHE[[i]]$tar)

}

train.OHE <- train.OHE[,-1]
test.OHE <- test.OHE[,-1]

rmv <- unique(which(is.na(test.OHE), arr.ind = TRUE)[,2])

if(length(rmv) != 0){
		train.OHE <- train.OHE[,-rmv]
		test.OHE <- test.OHE[,-rmv]
}

factors.OHE <- NULL


#Impute missing numeric values

print("imputing missing numeric values...", quote = FALSE)

train.new <- cbind(train[,numerics.index], train.OHE)
test.new <- cbind(test[,numerics.index], test.OHE)

n.trees <- floor(2*sqrt(nrow(train)))


cl <- makeCluster(detectCores() - 2) 
clusterEvalQ(cl, library(foreach))
registerDoParallel(cl)

train.imputation <- missForest(train.new, maxiter = 5, ntree = n.trees, parallelize = c("forests"), verbose = TRUE)

stopCluster(cl)

cl <- makeCluster(detectCores() - 2) 
clusterEvalQ(cl, library(foreach))
registerDoParallel(cl)

test.imputation <- missForest(test.new, maxiter = 5, ntree = n.trees, parallelize = c("forests"), verbose = TRUE)

stopCluster(cl)

train.out <- as.data.frame(train.imputation$ximp)
test.out <- as.data.frame(test.imputation$ximp)

if(scale.numerics){

		print("scaling numeric variables...", quote = FALSE)

		temp.frame <- train.out[,c(1:length(numerics.index))]

				cl <- makeCluster(detectCores()-2)
				registerDoParallel(cl)

				mu <- foreach(i = 1:length(numerics.index), .combine = "c") %dopar% mean(temp.frame[,i])

				stopCluster(cl)

				cl <- makeCluster(detectCores()-2)
				registerDoParallel(cl)

				sigma <- foreach(i = 1:length(numerics.index), .combine = "c") %dopar% mean(temp.frame[,i]^2)

				stopCluster(cl)				


sigma <- sqrt(sigma - mu^2)

for(i in 1:length(numerics.index)){

		train.out[,i] <- (train.out[,i] - mu[i]) / sigma[i]

		test.out[,i] <- (test.out[,i] - mu[i]) / sigma[i]
}

}



print("computing the linear model data frame...", quote = FALSE)

temp.frame <- train.out[,numerics.index]
temp.frame.2 <- test.out[,numerics.index]

Skews <- apply(temp.frame, 2, skewness)

to.fix <- which(Skews >= 0.75)

print(paste(length(to.fix), "highly skewed predictors were found..."), quote = FALSE)

if(length(to.fix) != 0){

	print("applying box-cox to highly skewed predictors of the linear model frame...", quote = FALSE)

				cl <- makeCluster(detectCores()-2)
				registerDoParallel(cl)

				lambdas <- foreach(i = 1:length(to.fix), .combine = "c", .packages = ("forecast")) %dopar% BoxCox.lambda(temp.frame[,to.fix[i]])

				stopCluster(cl)

				for(i in 1:length(to.fix)){

					temp.frame[,to.fix[i]] <- BoxCox(temp.frame[,to.fix[i]], lambda = lambdas[i])
					temp.frame.2[,to.fix[i]] <- BoxCox(temp.frame.2[,to.fix[i]], lambda = lambdas[i])

				}

}



print("checking if the column space basis is singular...", quote = FALSE)

train.L <- train.out
test.L <- test.out

train.L[,numerics.index] <- temp.frame
test.L[,numerics.index] <- temp.frame.2

train.L[,numerics.index] <- as.matrix(train.L[,numerics.index])
test.L[,numerics.index] <- as.matrix(test.L[,numerics.index])

colnames(train.L) <- colnames(train.out)
colnames(test.L) <- colnames(train.out)

temp.frame <- NULL
temp.frame.2 <- NULL

rmv <- findLinearCombos(train.out)$remove

if(length(rmv) != 0){

	print(paste("removing", length(rmv), "predictors to form a non-singular column space basis..."), quote = FALSE)

		train.L <- train.L[,-rmv]
		test.L <- test.L[,-rmv]

}



print("done.", quote = FALSE)
print(paste("#factors:", (ncol(train.out) - length(numerics.index))), quote = FALSE)
print(paste("#numerics:", length(numerics.index)), quote = FALSE)

target.var <- as.matrix(target.var)
colnames(target.var) <- "y"

l <- BoxCox.lambda(target.var)
target.var.L <- as.matrix(BoxCox(target.var, lambda = l))
colnames(target.var.L) <- "y"

end_time <- Sys.time()

print(proc.time() - ptm, quote = FALSE)

return(list(train.clean = train.out, test.clean = test.out, y = as.data.frame(target.var), train.clean.LINEAR = train.L, test.clean.LINEAR = test.L, y.L = as.data.frame(target.var.L), y.lambda = 1))

}