from typing import List
from typing import Tuple

"""
___author___: "Ryan Li Jian Tang"
___university___: "Monash University"

Brief:


Methodology:

"""


def best_itinerary(profit: List[List[int]], quarantine_time: List[int], home: int) -> int:
    """
    Finds highest possible money within the number of days given from the cities.

    :param profit: A list that contains the income possible for each day in a city
    :param quarantine_time: A list of non-negative integers, that represents the quarantine period of each city.
    :param home: A non-negative integer, that dictates from which city the person will begin from.
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
            memo_choice[day][city] = max(quarantine_value, optimal_value2, optimal_value3)    # Contains highest value you can earn on day, provided you just arrived at the city (Hence can only travel or quarantine)

    return memo_optimal[0][home]