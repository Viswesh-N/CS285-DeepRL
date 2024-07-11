import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse

def extract_data_from_event_file(file_path, tag):
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()

    timesteps = []
    values = []

    for event in ea.Scalars(tag):
        timesteps.append(event.step)
        values.append(event.value)

    return timesteps, values

def plot_experiments(data_dir, tag):
    plt.figure(figsize=(10, 6))

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                file_path = os.path.join(root, file)
                timesteps, values = extract_data_from_event_file(file_path, tag)
                label = os.path.basename(root)
                plt.plot(timesteps, values, label=label)

    plt.xlabel('Timesteps')
    plt.ylabel('Average Return' if tag == 'Eval_AverageReturn' else 'Critic Loss')
    plt.title(f'Average {"Eval Return" if tag == "Eval_AverageReturn" else "Critic Loss"} vs Timesteps for Multiple Experiments')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing event files')
    parser.add_argument('--critic_plot', action='store_true', help='Plot Critic Loss instead of Eval_AverageReturn')
    args = parser.parse_args()

    tag = 'Eval_AverageReturn' if not args.critic_plot else 'Critic Loss'
    plot_experiments(args.data_dir, tag)

if __name__ == "__main__":
    main()
