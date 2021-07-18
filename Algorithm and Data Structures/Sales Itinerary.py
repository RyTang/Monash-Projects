from typing import List
from typing import Tuple

"""
___author___: "Ryan Li Jian Tang"
___university___: "Monash University"

Brief:
This is related to a COVID-19 Scenario. A salesperson lives alongside a coast and travel along the coast to work in 
different cities. Sometimes they stay in a city for one day or multiple days to work, or just to pass by on their way to 
work in another city. The traveler knows the income he can get for each day (which changes everyday) in each city.

However since COVID-19, each city has their own unique quarantine period for travelers only if they wish to stay in the city.
If they're just passing by, they do not have to be quarantined. Travelers can either spend their day working, quarantining
or travelling to an adjacent city. Each city only has 2 adjacent cities, except the 2 cities at the end of the coast 
where they only have one.

The objective is to find the highest amount of income he can earn within a certain number of days.

Example of City layout

(One End) City 1 - City 2 - City 3 - City 4 (Other End)

City 2 is adjacent to both City 1 and City 3
City 1 and City 4 only have 1 adjacent city

Methodology:
The idea is to use top-down dynamic programming to solve the problem. We start the process from the last possible working
day. 

For each day a salesperson has 3 options:
Option 1: Quarantine if they just arrived and desire to work
Option 2: You have finished quarantine and can work.
Option 3: Travel to an adjacent city

Thus our algorithm will be based on these 3 options to fill the memo. 2 Memos will be used, one to keep track of the results
where a salesperson just arrived at a city and they can only either travel or quarantine. The other memo is to keep track
of the most optimal amount of income they can earn on a certain day in a city. The first memo is used to aid in the construction
of the 2nd memo. After the algorithm is done, the maximal income that can be earned for each city will be stored in the 
memo and be easily accessible.
"""


def best_itinerary(profit: List[List[int]], quarantine_time: List[int], home: int) -> int:
    """
    Finds highest possible money within the number of days given from the cities.

    :param profit: A list that contains the income possible for each day in a city
    :param quarantine_time: A list of non-negative integers, that represents the quarantine period of each city.
    :param home: A non-negative integer, that dictates from which city the traveler will begin from.
    :return: Highest possible amount of money that can be earned.
    :time_complexity: O(nd) where n is the number of cites and d is the number of days
    :space_complexity: O(nd) where n is the number of cites and d is the number of days
    """
    # local variables used
    num_days = len(profit)
    num_cities = len(quarantine_time)
    optimal_value1, optimal_value2, optimal_value3 = 0, 0, 0
    quarantine_value = 0

    if num_days <= 0 or num_cities <= 0:    # In the case where you can either earn no money or have no city to work in.
        return 0

    # Setting up memory
    memo_optimal = [[None for x in range(num_cities)] for j in range(num_days)]
    memo_choice = [[0 for x in range(num_cities)] for j in range(num_days)]
    memo_optimal[-1] = profit[-1]   # As the last day for each city is the optimal u can earn for that day

    for day in range(num_days - 2, -1, -1):  # Begin from 2nd last day to the very 1st day
        for city in range(num_cities):  # Go through each city for each day
            # For First Memory
            optimal_value1 = profit[day][city] + memo_optimal[day + 1][city]
            optimal_value2, optimal_value3 = 0, 0
            # For Second Memory
            quarantine_value = 0

            # Finding values
            if day + quarantine_time[city] < num_days:  # If you need to quarantine before working
                quarantine_value = memo_optimal[day + quarantine_time[city]][city]
            if city < num_cities - 1:  # Highest income you can get if you travel to city + 1
                optimal_value2 = memo_choice[day + 1][city + 1]
            if city > 0:  # Highest income you can get if you travel to city - 1
                optimal_value3 = memo_choice[day + 1][city - 1]

            # Fill in memory
            memo_optimal[day][city] = max(optimal_value1, optimal_value2, optimal_value3)   # Contains optimal income possible on a day and city
            memo_choice[day][city] = max(quarantine_value, optimal_value2, optimal_value3)    # Contains highest value you can earn on day, where you just arrived at the city (Hence can only travel or quarantine)

    return memo_optimal[0][home]
