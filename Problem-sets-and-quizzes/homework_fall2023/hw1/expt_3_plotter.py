import matplotlib.pyplot as plt
import numpy as np

# Data for multiple DAgger experiments
iterations = np.arange(10)

# Train steps 2000
eval_avg_returns_2000 = [
    1193.7120361328125, 2263.576416015625, 3127.911376953125, 3412.8291015625, 3713.5078125, 
    3732.673828125, 3723.245361328125, 3721.114501953125, 3725.036376953125, 3720.36474609375
]
eval_std_returns_2000 = [
    199.298828125, 1087.8018798828125, 837.1553955078125, 667.3729248046875, 4.594252109527588, 
    3.565241813659668, 3.138578414916992, 4.13463830947876, 3.6111507415771484, 2.7756776809692383
]

# Train steps 1500
eval_avg_returns_1500 = [
    957.1991577148438, 2193.900146484375, 3683.36083984375, 3700.498779296875, 3707.503173828125,
    3722.328857421875, 3723.100341796875, 3725.759033203125, 3717.44677734375, 3724.712890625
]
eval_std_returns_1500 = [
    460.6824035644531, 1071.1019287109375, 14.508522987365723, 7.686629772186279, 5.8469109535217285, 
    3.8139491081237793, 4.863668918609619, 3.3198413848876953, 3.2913002967834473, 4.973001480102539
]

# Train steps 1000
eval_avg_returns_1000 = [
    784.0283203125, 1382.2335205078125, 3521.603515625, 3137.237060546875, 3721.6328125, 
    3732.278076171875, 3728.46044921875, 3720.794189453125, 3722.594482421875, 3714.87646484375
]
eval_std_returns_1000 = [
    200.63255310058594, 409.114013671875, 79.96697998046875, 659.2938232421875, 3.7724838256835938, 
    3.3806889057159424, 4.210408687591553, 4.620635986328125, 4.194255828857422, 3.7965493202209473
]

# Train steps 500
eval_avg_returns_500 = [
    1047.47509765625, 937.5684814453125, 1771.0645751953125, 2274.1220703125, 3654.21142578125, 
    2751.09521484375, 3381.5009765625, 3683.91943359375, 3688.41259765625, 3707.15771484375
]
eval_std_returns_500 = [
    257.5317077636719, 84.9600830078125, 714.2741088867188, 902.9671630859375, 37.921199798583984, 
    831.3621826171875, 798.8222045898438, 11.845208168029785, 3.374000310897827, 2.6369309425354004
]

# Horizontal lines data
bc_avg_return = 784.0283203125
expert_avg_return = 3717.913330078125

# Create the plot
plt.figure(figsize=(12, 8))

# Plot each DAgger experiment with error bars
plt.errorbar(iterations, eval_avg_returns_2000, yerr=eval_std_returns_2000, fmt='-o', label='DAgger Policy (Train steps: 2000)')
plt.errorbar(iterations, eval_avg_returns_1500, yerr=eval_std_returns_1500, fmt='-o', label='DAgger Policy (Train steps: 1500)')
plt.errorbar(iterations, eval_avg_returns_1000, yerr=eval_std_returns_1000, fmt='-o', label='DAgger Policy (Train steps: 1000)')
plt.errorbar(iterations, eval_avg_returns_500, yerr=eval_std_returns_500, fmt='-o', label='DAgger Policy (Train steps: 500)')

# Add horizontal lines for BC agent and Expert policy
plt.axhline(y=bc_avg_return, color='r', linestyle='--', label='Behavioral Cloning Agent')
plt.axhline(y=expert_avg_return, color='g', linestyle='--', label='Expert Policy')

# Labels and title
plt.xlabel('DAgger Iterations')
plt.ylabel('Eval Average Return')
plt.title('DAgger Performance Over Iterations\nNetwork: 2-layer MLP, Task: Hopper-v4, Data: 10 iterations')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
