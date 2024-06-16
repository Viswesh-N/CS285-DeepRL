import matplotlib.pyplot as plt

# Data
train_steps = [500, 1000, 1500, 2000]
eval_avg_return = [4033.01, 4511.32, 4582.84, 4630.63]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_steps, eval_avg_return, marker='o', linestyle='-', color='b')
plt.xlabel('Train Steps')
plt.ylabel('Eval Average Return')
plt.title('Eval Average Return vs Train Steps')
plt.grid(True)
plt.annotate('A larger number of train steps leads to better fitting of the expert model',
             xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=10)
plt.show()
