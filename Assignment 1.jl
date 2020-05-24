using DataFrames
using CSV

using CSV
data = CSV.read("bank-additional-full.csv"; copycols=true)

showcols(data)

names(Data)

names(data)

show(data)

deleterows!(data,find(isunknown(data[:,symbol("education")])))

eachrow(data,find("unkown"))

filter!(row -> row[:job] != ("unknown"), data);

show(data)

filter!(row -> row[:marital] != ("unknown"), data);

show(data)

filter!(row -> row[:education] != ("unknown"), data);

show(data)

eachcol(filter!(row -> row[:marital] != ("unknown"), data);)

names(data)

filter!(row -> row[:default] != ("unknown"), data);

show(data)

tail(data)

filter!(row -> row[:housing] != ("unknown"), data);

show(data)

filter!(row -> row[:loan] != ("unknown"), data);

show(data)

filter!(row -> row[:contact] != ("unknown"), data);

show(data)

filter!(row -> row[:month] != ("unknown"), data);

filter!(row -> row[:day_of_week] != ("unknown"), data);

show(data)

names(data)

filter!(row -> row[Symbol("emp.var.rate")] != ("nonexistent"), data);

show(data)

[eltype(col) for col = eachcol(data)]

select!(data, Not([Symbol("emp.var.rate")]))

using StatsModels

Pkg.add("StatsModels")

StatsModels.ContrastsMatrix(DummyCoding(), [data]).matrix

using Pkg



Pkg.add("Pandas") 


using Pandas 

using Pkg
Pkg.add("Flux")
using Flux

using Flux: onehotbatch
hot = onehotbatch(:age, [:age, :job ,:marital,:education,:default,:housing,:loan                   
 ,:contact                
 ,:month                  
 ,:day_of_week            
 ,:duration               
 ,:campaign               
 ,:pdays                  
 ,:previous               
 ,:poutcome               
 ,Symbol("cons.price.idx")
 ,Symbol("cons.conf.idx") 
 ,:euribor3m              
 ,Symbol("nr.employed")   
 ,:y   ])

using DataFrames: CategoricalArrays

cv = CategoricalArray([data])

head(cv)

using DataFrames, CSV
using Plots, StatPlots
pyplot();


names(data)

@df data scatter(:job,:age, zcolor= :y, xaxis = "Job", yaxis="Age", lab="no")

head(data)

Pkg.add("ScikitLearn")
Pkg.update()

using ScikitLearn

datatest = data

datatest.y = replace(datatest.y, 1 => 0)

datatest.y = replace(datatest.y, "yes" => 1)

show(datatest, allrows=false, allcols=true)

datatest[!,:y] = convert.(Float64,datatest[!,:y])

datamatrix = convert(Matrix,datatest)

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

munet = datamatrix[:,end-2:end]

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

X = convert(Array,munet[:,[:1,:2]]);

Y = convert(Array,munet[:,[:3]]);

X

Y

head(data)

datatest[!,:y] = convert.(Float64,datatest[!,:y])

num_tr_ex = length(Y);

X = (X .- mean(X, dims = [99,1])) ./ std(X, dims = [99,1]);

size(X), size(Y)

W = param(zeros(2))
b = param([0.])

predict(x) = NNlib.σ.(x*W .+ b)

loss(x, y) = sum(binarycrossentropy.(predict(x), y))/num_tr_ex

using Flux: Params
par = Params([W, b])

#Pkg.add(Pkg.PackageSpec(;name="Flux", version="0.10"))
opt = SGD(par, 0.1; decay = 0)
evalcb() = @show(loss(X, Y))

cb = evalcb
datas = repeated((X, Y), 200)
Flux.train!(loss, datas, opt, cb)

opt = Descent(0.1) # Gradient descent with learning rate 0.1

for p in (W, b)
  update!(opt, p, grads[p])
end

data

datatest

 Pkg.add(Pkg.PackageSpec(;name="Flux", version="0.8.3"))

p = predict(X) .> 0.5
sum(p .== y) / length(y)

predict([0 0])


