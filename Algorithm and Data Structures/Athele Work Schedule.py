from typing import List
from typing import Tuple

"""
___author___: "Ryan Li Jian Tang"
___university___: "Monash University"

Brief:
The purpose of this assignment is to create a program that returns the highest possible income for an athlete. 
So the athlete mostly works as a personal trainer but also competes in sporting competitions where they earn a cash price
for participating. However, if the athlete joins a competition, they will need to spend time preparing and participating, thus,
being unable to earn cash from their personal trainer job.

Data Format:
weekly_income = [10, 5, 2] -> Shows there are 3 available working weeks from now. 1st week will earn 10 dollars, 2nd week 5 and so on.

Competitions = [(5, 10, 2), (1, 5, 100)] -> Each tuple is a competition available in the format of (start_week, end_week, winnings).
If the athlete chooses to participate in a competition, e.g. (1, 5, 10). That means they will be unavailable to work starting
from week 1 to week 5 but will earn 100 dollars from this competition.

Methodology:
To solve this, I used dynamic Programming. I first converted all given data into the same format. In this case, weekly_income
will be converted into the competitions format and stored together. Say for example if an athlete earns 10 dollars in week 1, 
the new format will then be (1, 1, 10). This is to allow easier processing and sorting. Then I can use dynamic programming
to run through each available choice and select the one with the highest income. 

Extra: 
Since this function only returns the highest income (which is not really informative), I can easily modify the function
to keep track of which particular activity was done for each week in the memo, at the expense of some memory. Then 
retrieving the weeks with the activities that will earn the highest income within the certain number of weeks.

"""


def best_schedule(weekly_income: List[int], competitions: List[Tuple[int, int, int]]) -> int:
    """
    Finds the highest possible income within the working weeks.

    :param weekly_income: A List of non-negative integers that represents the income that can be earned by working on that week.
    :param competitions: A List of non-negative Tuples that represents the competitions available in the format of (start_week, end_week, winnings).
    :return: The highest possible income within working weeks.
    :time_complexity: O(nlog(n)) where n is the total number of elements in both weekly_income and competitions.
    :space_complexity: O(n) where n is the total number of elements in both weekly_income and competitions.
    """
    memo = [0] * (len(weekly_income) + 1)  # memory for optimal solutions for available weeks
    # memo[0] will be the base case where there are no available income at all, so income = 0

    # Check if input is valid
    if len(weekly_income) > 0:
        # Change format of weekly_income into (start, end, income)
        new_ar = [(i, i, weekly_income[i]) for i in range(len(weekly_income))]
        # combine with available competitions
        for comp in competitions:
            new_ar.append(comp)

        # Then sort the new Combined list depending on end week using sort()
        new_ar.sort(key=lambda tup: tup[1])

        # Then use dp to search through to find most optimal answer for sup-problem starting from bottom up.
        for i in range(len(new_ar)):
            entry = new_ar[i]  # entry = (start, end, income)
            value = entry[-1] + memo[entry[0]]  # value = current income option + (optimal of the week before the job starts)
            if value > memo[entry[1] + 1]:  # if value > current highest possible optimal
                memo[entry[1] + 1] = value

    return memo[-1]
