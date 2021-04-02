from tqdm import trange
import numpy as np
import csv

data = []
res = []
with open('./x*3.csv', newline='') as csvfile:
    csvdata = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(csvdata)
    for row in csvdata:
        res.append(float(row[-1]))
        data.append([float(i) for i in row[:-1]])

fig = figure()
ax = fig.add_subplot(111, projection='3d')
for i in trange(len(data)):
    x = float(data[i][0])
    y = float(data[i][1])
    z = float(data[i][2])
    s = float(res[i]/100000)
    ax.scatter(x, y, z, s=s)
show()

#Linear Regression - Batch Gradient Descent
cost_function_results = []
def linear_regression(dataset, results, theta_js=None, lr=0.0001, its=10000):
    if theta_js is None:
        theta_js = np.random.rand(len(dataset[0])+1)
    for i in (t := trange(its)):
        cost_function_res = cost_function(theta_js, dataset, results)
        cost_function_results.append(cost_function_res)
        t.set_description('cost_fuction %.2f' % (cost_function_res))
        for theta_index in range(len(theta_js)):
            theta = theta_js[theta_index]
            derivitive_cost = derivitive_cost_function(theta_js, theta_index, dataset, results)
            theta_js[theta_index] = np.subtract(theta, np.multiply(lr, derivitive_cost))
        pass
    print(theta_js)
    return theta_js
        
def cost_function_piece(theta_js, x_js, result):
    h_theta = np.dot(theta_js, x_js)
    h_theta_sub_res = np.subtract(h_theta, result)
    return h_theta_sub_res

def derivitive_cost_function(theta_js, theta_index, dataset, results):
    derivitive_cost_pieces = []
    for data_index in range(len(dataset)):
        data = [1] + dataset[data_index]
        res = results[data_index]
        cost_piece = cost_function_piece(theta_js, data, res)
        derivitive_cost_pieces.append(np.multiply(cost_piece, data[theta_index]))
    return np.sum(derivitive_cost_pieces)

def cost_function(theta_js, dataset, results):
    cost_function_pieces = []
    for dataIndex in range(len(dataset)):
        data = [1] + dataset[dataIndex]
        result = results[dataIndex]
        cost_function_pieces.append(np.power(cost_function_piece(theta_js, data, result), 2))
    return np.divide(np.sum(cost_function_pieces), 2)

theta_res = linear_regression(data, res)
