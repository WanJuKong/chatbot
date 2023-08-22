from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
model = load_model('mnist_model.keras')
model.summary()
model.evaluate(x_test, y_test, verbose = 2)

plt.imshow(x_test[20], cmap = 'gray')
plt.show()

predict = model.predict(x_test[[20]])
for i in range(len(predict[0])):
    ac = predict[0][i] * 100
    print('{}:\t{}%'.format(i+1, f"{ac:.{7}f}"))
