# diamonds description
## Date: 2020-3-1
## About this file:

1. A data frame with 53940 rows and 10 variables:

- price price in US dollars (\$326--\$18,823)

- carat weight of the diamond (0.2--5.01)

- cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)

- color diamond colour, from J (worst) to D (best)

- clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

- x length in mm (0--10.74)

- y width in mm (0--58.9)

- z depth in mm (0--31.8)

- depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)

- table width of top of diamond relative to widest point (43--95)

Date:2020/4/12 
Author:Yu Huang
2. Columns

| Syntax      | Description |
| ----------- | ----------- |
| Index  | counter |
| carat  | Carat weight of the diamond |
| cut    | Describe cut quality of the diamond. Quality in increasing order Fair, Good, Very Good, Premium, Ideal |
| color  | Color of the diamond, with D being the best and J the worst |
| clarity| How obvious inclusions are within the diamond:(in order from best to worst, FL = flawless, I3= level 3 inclusions) FL,IF, VVS1, | VVS2, VS1, VS2, SI1, SI2, I1, I2, I3 |
| depth  | depth % :The height of a diamond, measured from the culet to the table, divided by its average girdle diameter |
| table  | table%: The width of the diamond's table expressed as a percentage of its average diameter |
| price  | the price of the diamond |
| x      | length mm |
| y      | width mm  |
| z      | depth mm  |
