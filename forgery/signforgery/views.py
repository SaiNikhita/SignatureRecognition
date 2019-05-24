from django.shortcuts import render
from django.http import HttpResponse
from django import forms
from signforgery.forms import UpdateForm
from .models import Image
import cv2
import numpy as np
import os
import random

class NeuralNetwork():

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def separate_batches(self, training_data, batch_size):
        random.shuffle(training_data)
        n = len(training_data)
        return [training_data[i:i + batch_size] for i in range(0, n, batch_size)]

    def update_batches(self, batches, alpha):
        for batch in batches:
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]

            m = len(batch)

            for x, y in batch:
                delta_b, delta_w = self.backpropagation(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]

            self.weights = [w - (alpha / m) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (alpha / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def sgd(self, training_data, epochs, batch_size, alpha, test_data):
        n_test = len(test_data)

        for epoch in range(epochs):
            batches = self.separate_batches(training_data, batch_size)
            self.update_batches(batches, alpha)

            print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        print(test_results)

        return sum(int(x == y) for (x, y) in test_results)

    def predict(self, training_data, batch_size, alpha, test_data):
        batches = self.separate_batches(training_data, batch_size)
        self.update_batches(batches, alpha)
        for x in test_data:
            result=(np.argmax(self.feedforward(x)))
            if result==1:
                return "genuine"
            else :
                return "forged"

    def cost_derivative(self, output_activations, y):
        return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def prepare(input):
    clean = cv2.fastNlMeansDenoising(input)
    ret, tresh = cv2.threshold(clean, 127, 1, cv2.THRESH_BINARY_INV)
    img = crop(tresh)

    flatten_img = cv2.resize(img, (40, 10), interpolation=cv2.INTER_AREA).flatten()

    resized = cv2.resize(img, (400, 100), interpolation=cv2.INTER_AREA)
    columns = np.sum(resized, axis=0)  # sum of all columns
    lines = np.sum(resized, axis=1)  # sum of all lines

    h, w = img.shape
    aspect = w / h

    return [*flatten_img, *columns, *lines, aspect]


def crop(img):
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y + h, x: x + w]

def home(request):
    if request.method == 'POST':
        form = UpdateForm(request.POST, request.FILES, initial={'classified': 'genuine'})
        if form.is_valid():
            form.save()
            obj = Image.objects.last()
            all = Image.objects.filter(name=obj.name).exclude(imagefile=obj.imagefile)
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            MEDIA_ROOT = os.path.join(BASE_DIR, 'media/')
            training_data = []
            for filename in all:
                img = cv2.imread(str(MEDIA_ROOT+ str(filename.imagefile)), 0)
                if img is not None:
                    data = np.array(prepare(img))
                    data = np.reshape(data, (901, 1))
                    if str(filename.classified) == "genuine":
                        result = [[0], [1]]
                    else:
                        result= [[1], [0]]
                    result = np.array(result)
                    result = np.reshape(result, (2, 1))
                    training_data.append((data, result))
            net = NeuralNetwork([901, 500, 500, 2])
            predict = []
            img = cv2.imread(str(MEDIA_ROOT+ str(obj.imagefile)), 0)
            if img is not None:
                data = np.array(prepare(img))
                data = np.reshape(data, (901, 1))
                predict.append((data))
            classy = net.predict(training_data,50, 0.01,predict)
            obj.classified = classy
            obj.save()
            return HttpResponse(classy)

    else:
        form = UpdateForm()
        return render(request, "neural.html", {'form': form})
