import math

# Given values
A_x, A_y = 39, 98
AB_length = 518
theta = math.radians(49)  # Convert degrees to radians for calculation

# Calculate the change in x and y
delta_x = AB_length * math.cos(theta)
delta_y = AB_length * math.sin(theta)

# Calculate B's coordinates
B_x = A_x + delta_x
B_y = A_y + delta_y

B_x, B_y
print(B_x, B_y)