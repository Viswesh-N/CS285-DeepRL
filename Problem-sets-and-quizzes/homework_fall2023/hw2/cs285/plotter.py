import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse

def extract_data_from_event_file(file_path):
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()

    timesteps = []
    avg_returns = []

    for event in ea.Scalars('Eval_AverageReturn'):
        timesteps.append(event.step)
        avg_returns.append(event.value)

    return timesteps, avg_returns

def plot_experiments(data_dir):
    plt.figure(figsize=(10, 6))

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                file_path = os.path.join(root, file)
                timesteps, avg_returns = extract_data_from_event_file(file_path)
                label = os.path.basename(root)
                plt.plot(timesteps, avg_returns, label=label)

    plt.xlabel('Timesteps')
    plt.ylabel('Average Return')
    plt.title('Average Eval Return vs Timesteps for Multiple Experiments')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing event files')
    args = parser.parse_args()

    plot_experiments(args.data_dir)

if __name__ == "__main__":
    main()
