# Macroelement model
In this project a macroelement model to represent foundations was developed.

As a brief introduction, a macroelement model is a physical model that allows to 
reproduce linear and non-linear behaviour of real life systems in a simplified way.

In the foundations case, a macroelement model tries to reproduce the Force-Displacement 
behaviour of foundations in a simplified manner. It consist of three main components; 
the constitutive relation, the non linearity function and the potential function.
A constitutive relation is an equation that describes the relation between forces and displacements for the macroelement case.
The non linearity constant controls the shape of the produced curve in the Force-Displacement space.
The potential function is added to describe the direction of displacements produced by an increment in force,
in this way we can accurately describe the displacements of foundations subjected to different types of load paths.

The main archive, called Macroelement_driver.py consists of the full implementation of a hypoplastic macroelement 
with a separate model for uplift of foundations, it can be activated and deactivated by changing the input parameter "uplift_code" 
from "Active" to "non-active". This mechanism allows to reproduce uplift of shallow foundations in a better way.

The hypoplastic model is implemented in a Runge-Kutta Fehlberg integration scheme and its optimized to work with numba jit for smooth calculations.
