# Lab2

## Performance Distributions

The MPG (miles per gallon) performance of vehicles in each fleet were first visualised.
This gives a general view of the distribution of MPG performance amongst each fleet, as a basis for comparison

* For the existing fleet :
![logo](./histogram_current.png?raw=true)

* For the new proposed fleet :
![logo](./histogram_new.png?raw=true)

## Standard Deviations (bootstrapped)

The statistical range of MPG within each fleet was assessed, using resampling with replacement.
This gives the best approximation of a larger data set, to represent real-world behaviour

For the existing fleet :
(note this chart is Std Dev, not Mean)
![logo](./bootstrap_confidence_current.png?raw=true)

For the new proposed fleet :
![logo](./bootstrap_confidence_new.png?raw=true)
