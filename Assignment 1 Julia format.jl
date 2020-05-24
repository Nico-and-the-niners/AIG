using DataFrames
using CSV

#Loading Data
using CSV
data = CSV.read("bank-additional-full.csv"; copycols=true)

#checking all the columns
names(data)

show(data)

#filter out the unkown data for jobs
filter!(row -> row[:job] != ("unknown"), data);

show(data)

#filter out unkowns for marital
filter!(row -> row[:marital] != ("unknown"), data);

show(data)

#filter out education unknown
filter!(row -> row[:education] != ("unknown"), data);

show(data)

eachcol(filter!(row -> row[:marital] != ("unknown"), data);)

#filter out unkowns for defualt
filter!(row -> row[:default] != ("unknown"), data);

show(data)

tail(data)

show(data)

#filter out unknowns for loan
filter!(row -> row[:loan] != ("unknown"), data);

show(data)

filter!(row -> row[:contact] != ("unknown"), data);

show(data)

filter!(row -> row[:month] != ("unknown"), data);

filter!(row -> row[:day_of_week] != ("unknown"), data);

filter!(row -> row[Symbol("emp.var.rate")] != ("nonexistent"), data);

#check column types
[eltype(col) for col = eachcol(data)]

#remove column with too much nonexixtent data
select!(data, Not([Symbol("emp.var.rate")]))

using StatsModels

#enable the flux framework
using Pkg
Pkg.add("Flux")
using Flux

using DataFrames: CategoricalArrays

#assinged the data to variable for insurance 
datatest = data

#changing the target variable to binary
datatest.y = replace(datatest.y, "yes" => 1)

datatest.y = replace(datatest.y, "no" => 0)

show(datatest, allrows=false, allcols=true)

#changing target into float for model
datatest[!,:y] = convert.(Float64,datatest[!,:y])

#converting data into a matrix
datamatrix = convert(Matrix,datatest)

#iterating through matrix
size(datamatrix)[2]
for i = 1:(size(datamatrix)[2]-2)
    uniquee = unique(datamatrix[:,i])
    for j = 1:size(datamatrix)[1]
        for k = 1:size(uniquee)[1]
            if datamatrix[j,i] == uniquee[k]
                datamatrix[j,i] = Int(k)
            end
        end
    end
    println(i)
end

munet = datamatrix[:,end-4:end]

using Flux
using Statistics

using Flux.NNlib
using Flux: throttle, normalise
using Flux: binarycrossentropy
using Base.Iterators: repeated
using Flux: @epochs

munet

using NamedArrays
test = NamedArray(munet)

#dividing features into X variable and and target into Y
X = convert(Array,munet[:,[:1,:2,:3,:4]]);

Y = convert(Array,munet[:,[:5]]);

X

Y

#converting Y set into float for training
datatest[!,:y] = convert.(Float64,datatest[!,:y])

num_tr_ex = length(Y);
num_tr_ex

#naormalizing some features
X = (X .- mean(X, dims = [90,1])) ./ std(X, dims = [90,1]);

size(X), size(Y)

#Here we build a model for Log Regression in flux
#param tells it what to train
W = param(zeros(2))
b = param([0.])

#W is matrix of weight and b is bias
predict(x) = NNlib.σ.(x*W .+ b)

#here we setup loss function
loss(x, y) = sum(binarycrossentropy.(predict(x), y))/num_tr_ex

using Flux: Params
par = Params([W, b])

#Here we using Stockastic descent with learning rate of 0.01
using Flux: Params
par = Params([W, b])
opt = SGD(par, 0.1; decay = 0)
evalcb() = @show(loss(X, Y))

#here we train the model
cb = evalcb
datas = repeated((X, Y), 200)
Flux.train!(loss, datas, opt, cb)

#We test the accuracy
p = predict(x) .> 0.5
sum(p .== y) / length(Y)

predict([0 1])


