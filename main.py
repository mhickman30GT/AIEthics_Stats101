import datetime
import os
import copy
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# GLOBAL VARIABLES
TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PATH = os.path.dirname(os.path.realpath(__file__))


class DataSet:
    """ Class holding values for dataset """

    def __init__(self, file, direct):
        """ Constructor for Dataset """
        self.data_name = f'Deaths In Custody Data Set'
        self.out_dir = direct
        self.file = file
        self.csv = pd.read_csv(self.file)
        self.reduced = pd.DataFrame()
        self.x_data = pd.DataFrame()
        self.race = None
        self.gender = None
        self.age = None
        self.manner = None
        self.custody = None
        self.features = list()

    def process(self):
        """ Processes data set """
        # Separate classification labels
        self.x_data = self.csv.drop(["manner_of_death", "custody_status"], 1)
        self.manner = copy.deepcopy(self.csv["manner_of_death"])
        self.custody = copy.deepcopy(self.csv["custody_status"])
        self.race = copy.deepcopy(self.csv["race"])
        self.gender = copy.deepcopy(self.csv["gender"])
        self.age = copy.deepcopy(self.csv["age"])
        self.features = list(self.x_data.columns)

        # Bin Race to combine Asian/Oceanic:
        for index, race in self.csv["race"].items():
            if (race == "Other Asian" or race == "Filipino" or race == "Vietnamese" or
                    race == "Asian Indian" or race == "Pacific Islander" or race == "Korean" or
                    race == "Chinese" or race == "Laotian" or race == "Samoan" or race == "Cambodian" or
                    race == "Japanese" or race == "Hawaiian" or race == "Guamanian"):

                # Set all these to Asian/Oceanic
                self.race[index] = "Asian/Oceanic"

        # Bin Age into relevant decades
        for index, age in self.csv["age"].items():
            if age == "Unk":
                self.age[index] = "Unknown"
            elif int(age) <= 29:
                self.age[index] = "0-29"
            elif int(age) <= 39:
                self.age[index] = "30-39"
            elif int(age) <= 49:
                self.age[index] = "40-49"
            elif int(age) <= 59:
                self.age[index] = "50-59"
            elif int(age) <= 69:
                self.age[index] = "60-69"
            else:
                self.age[index] = "70+"

    def reduce(self):
        """ Reduces data set by 50% via random sampling """
        # Reduce via random sample with no duplicates
        self.reduced = self.csv.sample(frac=0.5, replace=False, random_state=1)
        reduced_copy = copy.deepcopy(self)
        reduced_copy.csv = self.reduced
        reduced_copy.process()

        return reduced_copy

    def generate_race_hist(self):
        """ Generate histogram for race variable """
        # Create dicts for histograms
        wh_ma_dict = dict({'Natural': 0, 'Accidental': 0, 'Suicide': 0, 'Cannot be Determined': 0,
                           'Homicide Willful (Other Inmate)': 0, 'Homicide Justified (Law Enforcement Staff)': 0,
                           'Other': 0, 'Homicide Willful (Law Enforcement Staff)': 0, 'Execution': 0,
                           'Pending Investigation': 0, 'Homicide Justified (Other Inmate)' : 0})
        wh_cs_dict = dict({'Sentenced': 0, 'Process of Arrest': 0, 'Booked - Awaiting Trial': 0,
                           'Booked - No Charges Filed': 0, 'Awaiting Booking': 0, 'Other': 0, 'In Transit': 0,
                           'Out to Court': 0})
        hi_ma_dict = copy.deepcopy(wh_ma_dict)
        hi_cs_dict = copy.deepcopy(wh_cs_dict)
        bl_ma_dict = copy.deepcopy(wh_ma_dict)
        bl_cs_dict = copy.deepcopy(wh_cs_dict)
        as_ma_dict = copy.deepcopy(wh_ma_dict)
        as_cs_dict = copy.deepcopy(wh_cs_dict)
        ot_ma_dict = copy.deepcopy(wh_ma_dict)
        ot_cs_dict = copy.deepcopy(wh_cs_dict)
        am_ma_dict = copy.deepcopy(wh_ma_dict)
        am_cs_dict = copy.deepcopy(wh_cs_dict)

        # Find independent results for each
        for index, race in self.race.items():
            if race == "White":
                # Parse Manner of Death
                if self.manner[index] in wh_ma_dict.keys():
                    wh_ma_dict[self.manner[index]] += 1
                else:
                    wh_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in wh_cs_dict.keys():
                    wh_cs_dict[self.custody[index]] += 1
                else:
                    wh_cs_dict[self.custody[index]] = 1

            elif race == "Hispanic":
                # Parse Manner of Death
                if self.manner[index] in hi_ma_dict.keys():
                    hi_ma_dict[self.manner[index]] += 1
                else:
                    hi_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in hi_cs_dict.keys():
                    hi_cs_dict[self.custody[index]] += 1
                else:
                    hi_cs_dict[self.custody[index]] = 1

            elif race == "Black":
                # Parse Manner of Death
                if self.manner[index] in bl_ma_dict.keys():
                    bl_ma_dict[self.manner[index]] += 1
                else:
                    bl_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in bl_cs_dict.keys():
                    bl_cs_dict[self.custody[index]] += 1
                else:
                    bl_cs_dict[self.custody[index]] = 1

            elif race == "Asian/Oceanic":
                # Parse Manner of Death
                if self.manner[index] in as_ma_dict.keys():
                    as_ma_dict[self.manner[index]] += 1
                else:
                    as_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in as_cs_dict.keys():
                    as_cs_dict[self.custody[index]] += 1
                else:
                    as_cs_dict[self.custody[index]] = 1

            elif race == "Other":
                # Parse Manner of Death
                if self.manner[index] in ot_ma_dict.keys():
                    ot_ma_dict[self.manner[index]] += 1
                else:
                    ot_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in ot_cs_dict.keys():
                    ot_cs_dict[self.custody[index]] += 1
                else:
                    ot_cs_dict[self.custody[index]] = 1

            elif race == "American Indian":
                # Parse Manner of Death
                if self.manner[index] in am_ma_dict.keys():
                    am_ma_dict[self.manner[index]] += 1
                else:
                    am_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in am_cs_dict.keys():
                    am_cs_dict[self.custody[index]] += 1
                else:
                    am_cs_dict[self.custody[index]] = 1

        # Generate the graphs
        #print(wh_ma_dict)
        #print(hi_ma_dict)
        #print(bl_ma_dict)
        #print(as_ma_dict)
        #print(ot_ma_dict)
        #print(am_ma_dict)

        x_labels = ['White', 'Hispanic', 'Black', 'Asian/Oceanic', 'Other', 'American Indian']
        current = 'Natural'
        natural = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                   as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]
        current = 'Homicide Justified (Law Enforcement Staff)'
        homi_just = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                     as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]
        current = 'Suicide'
        suicide = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                   as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]
        current = 'Accidental'
        accident = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                    as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]
        current = 'Pending Investigation'
        pending = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                   as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]
        current = 'Homicide Willful (Other Inmate)'
        homi_will = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                     as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]
        current = 'Cannot be Determined'
        unknown = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                   as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]
        current = 'Other'
        other = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                 as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]
        current = 'Homicide Willful (Law Enforcement Staff)'
        homi_law = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                    as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]
        current = 'Homicide Justified (Other Inmate)'
        homi_inm = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                    as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]
        current = 'Execution'
        execution = [wh_ma_dict[current], hi_ma_dict[current], bl_ma_dict[current],
                     as_ma_dict[current], ot_ma_dict[current], am_ma_dict[current]]

        x_axis = np.arange(len(x_labels))

        fig, ax = plt.subplots()

        plt.bar(x_axis - 0.25, natural, 0.05, label='Natural')
        plt.bar(x_axis + 0.25, homi_just, 0.05, label='Homicide Justified: Law Enforcement')
        plt.bar(x_axis - 0.2, suicide, 0.05, label='Suicide')
        plt.bar(x_axis + 0.2, accident, 0.05, label='Accidental')
        plt.bar(x_axis - 0.15, pending, 0.05, label='Pending Investigation')
        plt.bar(x_axis + 0.15, homi_will, 0.05, label='Homicide Willful: Inmate')
        plt.bar(x_axis - 0.1, unknown, 0.05, label='Unknown')
        plt.bar(x_axis + 0.1, other, 0.05, label='Other')
        plt.bar(x_axis - 0.05, homi_law, 0.05, label='Homicide Willful: Law Enforcement')
        plt.bar(x_axis + 0.05, homi_inm, 0.05, label='Homicide Justified: Inmate')
        plt.bar(x_axis + 0.0, execution, 0.05, label='Execution')

        plt.xticks(x_axis, x_labels)
        plt.xlabel("Racial Labels")
        plt.ylabel("Number of Deaths")
        plt.title("Number of Deaths per Racial Label by Manner of Death")
        ax.xaxis_date()
        ax.autoscale(tight=True)
        plt.legend()
        fig.savefig(os.path.join(self.out_dir, f"race_hist_manner.png"))

        # Generate the graphs
        #print(wh_cs_dict)
        #print(hi_cs_dict)
        #print(bl_cs_dict)
        #print(as_cs_dict)
        #print(ot_cs_dict)
        #print(am_cs_dict)

        current = 'Sentenced'
        sentenced = [wh_cs_dict[current], hi_cs_dict[current], bl_cs_dict[current],
                     as_cs_dict[current], ot_cs_dict[current], am_cs_dict[current]]
        current = 'Process of Arrest'
        process_of = [wh_cs_dict[current], hi_cs_dict[current], bl_cs_dict[current],
                      as_cs_dict[current], ot_cs_dict[current], am_cs_dict[current]]
        current = 'Booked - Awaiting Trial'
        booked_await = [wh_cs_dict[current], hi_cs_dict[current], bl_cs_dict[current],
                        as_cs_dict[current], ot_cs_dict[current], am_cs_dict[current]]
        current = 'Other'
        other = [wh_cs_dict[current], hi_cs_dict[current], bl_cs_dict[current],
                 as_cs_dict[current], ot_cs_dict[current], am_cs_dict[current]]
        current = 'Booked - No Charges Filed'
        booked_no = [wh_cs_dict[current], hi_cs_dict[current], bl_cs_dict[current],
                     as_cs_dict[current], ot_cs_dict[current], am_cs_dict[current]]
        current = 'Awaiting Booking'
        awaiting = [wh_cs_dict[current], hi_cs_dict[current], bl_cs_dict[current],
                    as_cs_dict[current], ot_cs_dict[current], am_cs_dict[current]]
        current = 'In Transit'
        transit = [wh_cs_dict[current], hi_cs_dict[current], bl_cs_dict[current],
                   as_cs_dict[current], ot_cs_dict[current], am_cs_dict[current]]
        current = 'Out to Court'
        outtocourt = [wh_cs_dict[current], hi_cs_dict[current], bl_cs_dict[current],
                      as_cs_dict[current], ot_cs_dict[current], am_cs_dict[current]]

        x_axis = np.arange(len(x_labels))

        fig, ax = plt.subplots()

        plt.bar(x_axis - 0.2, sentenced, 0.05, label='Sentenced')
        plt.bar(x_axis + 0.15, process_of, 0.05, label='Process of Arrest')
        plt.bar(x_axis - 0.15, booked_await, 0.05, label='Booked - Awaiting Trial')
        plt.bar(x_axis + 0.1, other, 0.05, label='Other')
        plt.bar(x_axis - 0.1, booked_no, 0.05, label='Booked - No Charges Filed')
        plt.bar(x_axis + 0.05, awaiting, 0.05, label='Awaiting Booking')
        plt.bar(x_axis - 0.05, transit, 0.05, label='In Transit')
        plt.bar(x_axis + 0.0, outtocourt, 0.05, label='Out to Court')

        plt.xticks(x_axis, x_labels)
        plt.xlabel("Racial Labels")
        plt.ylabel("Number of Deaths")
        plt.title("Number of Deaths per Racial Label by Custody Status")
        ax.xaxis_date()
        ax.autoscale(tight=True)
        plt.legend()
        fig.savefig(os.path.join(self.out_dir, f"race_hist_custody.png"))

    def generate_age_hist(self):
        """ Generate histogram for age variable """
        # Create dicts for histograms
        zero_ma_dict = dict({'Natural': 0, 'Accidental': 0, 'Suicide': 0, 'Cannot be Determined': 0,
                             'Homicide Willful (Other Inmate)': 0, 'Homicide Justified (Law Enforcement Staff)': 0,
                             'Other': 0, 'Homicide Willful (Law Enforcement Staff)': 0, 'Execution': 0,
                             'Pending Investigation': 0, 'Homicide Justified (Other Inmate)' : 0})
        zero_cs_dict = dict({'Sentenced': 0, 'Process of Arrest': 0, 'Booked - Awaiting Trial': 0,
                             'Booked - No Charges Filed': 0, 'Awaiting Booking': 0, 'Other': 0, 'In Transit': 0,
                             'Out to Court': 0})
        three_ma_dict = copy.deepcopy(zero_ma_dict)
        three_cs_dict = copy.deepcopy(zero_cs_dict)
        four_ma_dict = copy.deepcopy(zero_ma_dict)
        four_cs_dict = copy.deepcopy(zero_cs_dict)
        five_ma_dict = copy.deepcopy(zero_ma_dict)
        five_cs_dict = copy.deepcopy(zero_cs_dict)
        six_ma_dict = copy.deepcopy(zero_ma_dict)
        six_cs_dict = copy.deepcopy(zero_cs_dict)
        seven_ma_dict = copy.deepcopy(zero_ma_dict)
        seven_cs_dict = copy.deepcopy(zero_cs_dict)

        # Find independent results for each
        for index, age in self.age.items():
            if age == "0-29":
                # Parse Manner of Death
                if self.manner[index] in zero_ma_dict.keys():
                    zero_ma_dict[self.manner[index]] += 1
                else:
                    zero_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in zero_cs_dict.keys():
                    zero_cs_dict[self.custody[index]] += 1
                else:
                    zero_cs_dict[self.custody[index]] = 1

            elif age == "30-39":
                # Parse Manner of Death
                if self.manner[index] in three_ma_dict.keys():
                    three_ma_dict[self.manner[index]] += 1
                else:
                    three_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in three_cs_dict.keys():
                    three_cs_dict[self.custody[index]] += 1
                else:
                    three_cs_dict[self.custody[index]] = 1

            elif age == "40-49":
                # Parse Manner of Death
                if self.manner[index] in four_ma_dict.keys():
                    four_ma_dict[self.manner[index]] += 1
                else:
                    four_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in four_cs_dict.keys():
                    four_cs_dict[self.custody[index]] += 1
                else:
                    four_cs_dict[self.custody[index]] = 1

            elif age == "50-59":
                # Parse Manner of Death
                if self.manner[index] in five_ma_dict.keys():
                    five_ma_dict[self.manner[index]] += 1
                else:
                    five_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in five_cs_dict.keys():
                    five_cs_dict[self.custody[index]] += 1
                else:
                    five_cs_dict[self.custody[index]] = 1

            elif age == "60-69":
                # Parse Manner of Death
                if self.manner[index] in six_ma_dict.keys():
                    six_ma_dict[self.manner[index]] += 1
                else:
                    six_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in six_cs_dict.keys():
                    six_cs_dict[self.custody[index]] += 1
                else:
                    six_cs_dict[self.custody[index]] = 1

            elif age == "70+":
                # Parse Manner of Death
                if self.manner[index] in seven_ma_dict.keys():
                    seven_ma_dict[self.manner[index]] += 1
                else:
                    seven_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in seven_cs_dict.keys():
                    seven_cs_dict[self.custody[index]] += 1
                else:
                    seven_cs_dict[self.custody[index]] = 1

        # Generate the graphs
        #print(zero_ma_dict)
        #print(three_ma_dict)
        #print(four_ma_dict)
        #print(five_ma_dict)
        #print(six_ma_dict)
        #print(seven_ma_dict)

        x_labels = ['0-29', '30-39', '40-49', '50-59', '60-69', '70+']
        current = 'Natural'
        natural = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                   five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]
        current = 'Homicide Justified (Law Enforcement Staff)'
        homi_just = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                     five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]
        current = 'Suicide'
        suicide = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                   five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]
        current = 'Accidental'
        accident = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                    five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]
        current = 'Pending Investigation'
        pending = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                   five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]
        current = 'Homicide Willful (Other Inmate)'
        homi_will = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                     five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]
        current = 'Cannot be Determined'
        unknown = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                   five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]
        current = 'Other'
        other = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                 five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]
        current = 'Homicide Willful (Law Enforcement Staff)'
        homi_law = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                    five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]
        current = 'Homicide Justified (Other Inmate)'
        homi_inm = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                    five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]
        current = 'Execution'
        execution = [zero_ma_dict[current], three_ma_dict[current], four_ma_dict[current],
                     five_ma_dict[current], six_ma_dict[current], seven_ma_dict[current]]

        x_axis = np.arange(len(x_labels))

        fig, ax = plt.subplots()

        plt.bar(x_axis - 0.25, natural, 0.05, label='Natural')
        plt.bar(x_axis + 0.25, homi_just, 0.05, label='Homicide Justified: Law Enforcement')
        plt.bar(x_axis - 0.2, suicide, 0.05, label='Suicide')
        plt.bar(x_axis + 0.2, accident, 0.05, label='Accidental')
        plt.bar(x_axis - 0.15, pending, 0.05, label='Pending Investigation')
        plt.bar(x_axis + 0.15, homi_will, 0.05, label='Homicide Willful: Inmate')
        plt.bar(x_axis - 0.1, unknown, 0.05, label='Unknown')
        plt.bar(x_axis + 0.1, other, 0.05, label='Other')
        plt.bar(x_axis - 0.05, homi_law, 0.05, label='Homicide Willful: Law Enforcement')
        plt.bar(x_axis + 0.05, homi_inm, 0.05, label='Homicide Justified: Inmate')
        plt.bar(x_axis + 0.0, execution, 0.05, label='Execution')

        plt.xticks(x_axis, x_labels)
        plt.xlabel("Age Labels")
        plt.ylabel("Number of Deaths")
        plt.title("Number of Deaths per Age Group by Manner of Death")
        ax.xaxis_date()
        ax.autoscale(tight=True)
        plt.legend()
        fig.savefig(os.path.join(self.out_dir, f"age_hist_manner.png"))

        # Generate the graphs
        #print("----------------------------------------------------------------------")
        #print(zero_cs_dict)
        #print(three_cs_dict)
        #print(four_cs_dict)
        #print(five_cs_dict)
        #print(six_cs_dict)
        #print(seven_cs_dict)

        current = 'Sentenced'
        sentenced = [zero_cs_dict[current], three_cs_dict[current], four_cs_dict[current],
                     five_cs_dict[current], six_cs_dict[current], seven_cs_dict[current]]
        current = 'Process of Arrest'
        proc = [zero_cs_dict[current], three_cs_dict[current], four_cs_dict[current],
                five_cs_dict[current], six_cs_dict[current], seven_cs_dict[current]]
        current = 'Booked - Awaiting Trial'
        booked_aw = [zero_cs_dict[current], three_cs_dict[current], four_cs_dict[current],
                     five_cs_dict[current], six_cs_dict[current], seven_cs_dict[current]]
        current = 'Booked - No Charges Filed'
        booked_no = [zero_cs_dict[current], three_cs_dict[current], four_cs_dict[current],
                     five_cs_dict[current], six_cs_dict[current], seven_cs_dict[current]]
        current = 'Awaiting Booking'
        awaitb = [zero_cs_dict[current], three_cs_dict[current], four_cs_dict[current],
                  five_cs_dict[current], six_cs_dict[current], seven_cs_dict[current]]
        current = 'Other'
        othe = [zero_cs_dict[current], three_cs_dict[current], four_cs_dict[current],
                five_cs_dict[current], six_cs_dict[current], seven_cs_dict[current]]
        current = 'In Transit'
        trans = [zero_cs_dict[current], three_cs_dict[current], four_cs_dict[current],
                 five_cs_dict[current], six_cs_dict[current], seven_cs_dict[current]]
        current = 'Out to Court'
        court = [zero_cs_dict[current], three_cs_dict[current], four_cs_dict[current],
                 five_cs_dict[current], six_cs_dict[current], seven_cs_dict[current]]

        x_axis = np.arange(len(x_labels))

        fig, ax = plt.subplots()

        plt.bar(x_axis - 0.2, sentenced, 0.05, label='Sentenced')
        plt.bar(x_axis + 0.15, proc, 0.05, label='Process of Arrest')
        plt.bar(x_axis - 0.15, booked_aw, 0.05, label='Booked - Awaiting Trial')
        plt.bar(x_axis + 0.1, booked_no, 0.05, label='Booked - No Charges Filed')
        plt.bar(x_axis - 0.1, awaitb, 0.05, label='Awaiting Booking')
        plt.bar(x_axis + 0.05, othe, 0.05, label='Other')
        plt.bar(x_axis - 0.05, trans, 0.05, label='In Transit')
        plt.bar(x_axis + 0.0, court, 0.05, label='Out to Court')

        plt.xticks(x_axis, x_labels)
        plt.xlabel("Age Labels")
        plt.ylabel("Number of Deaths")
        plt.title("Number of Deaths per Age Group by Custody Status")
        ax.xaxis_date()
        ax.autoscale(tight=True)
        plt.legend()
        fig.savefig(os.path.join(self.out_dir, f"age_hist_custody.png"))

    def generate_gender_hist(self):
        """ Generate histogram for age variable """
        # Create dicts for histograms
        f_ma_dict = dict({'Natural': 0, 'Accidental': 0, 'Suicide': 0, 'Cannot be Determined': 0,
                          'Homicide Willful (Other Inmate)': 0, 'Homicide Justified (Law Enforcement Staff)': 0,
                          'Other': 0, 'Homicide Willful (Law Enforcement Staff)': 0, 'Execution': 0,
                          'Pending Investigation': 0, 'Homicide Justified (Other Inmate)' : 0})
        f_cs_dict = dict({'Sentenced': 0, 'Process of Arrest': 0, 'Booked - Awaiting Trial': 0,
                          'Booked - No Charges Filed': 0, 'Awaiting Booking': 0, 'Other': 0, 'In Transit': 0,
                          'Out to Court': 0})
        m_ma_dict = copy.deepcopy(f_ma_dict)
        m_cs_dict = copy.deepcopy(f_cs_dict)

        # Find independent results for each
        for index, gender in self.gender.items():
            if gender == "Male":
                # Parse Manner of Death
                if self.manner[index] in m_ma_dict.keys():
                    m_ma_dict[self.manner[index]] += 1
                else:
                    m_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in m_cs_dict.keys():
                    m_cs_dict[self.custody[index]] += 1
                else:
                    m_cs_dict[self.custody[index]] = 1

            elif gender == "Female":
                # Parse Manner of Death
                if self.manner[index] in f_ma_dict.keys():
                    f_ma_dict[self.manner[index]] += 1
                else:
                    f_ma_dict[self.manner[index]] = 1
                # Parse Custody Status
                if self.custody[index] in f_cs_dict.keys():
                    f_cs_dict[self.custody[index]] += 1
                else:
                    f_cs_dict[self.custody[index]] = 1

        # Generate the graphs
        #print(m_ma_dict)
        #print(f_ma_dict)

        x_labels = ['Male', 'Female']
        current = 'Natural'
        natural = [m_ma_dict[current], f_ma_dict[current]]
        current = 'Homicide Justified (Law Enforcement Staff)'
        homi_just = [m_ma_dict[current], f_ma_dict[current]]
        current = 'Suicide'
        suicide = [m_ma_dict[current], f_ma_dict[current]]
        current = 'Accidental'
        accident = [m_ma_dict[current], f_ma_dict[current]]
        current = 'Pending Investigation'
        pending = [m_ma_dict[current], f_ma_dict[current]]
        current = 'Homicide Willful (Other Inmate)'
        homi_will = [m_ma_dict[current], f_ma_dict[current]]
        current = 'Cannot be Determined'
        unknown = [m_ma_dict[current], f_ma_dict[current]]
        current = 'Other'
        other = [m_ma_dict[current], f_ma_dict[current]]
        current = 'Homicide Willful (Law Enforcement Staff)'
        homi_law = [m_ma_dict[current], f_ma_dict[current]]
        current = 'Homicide Justified (Other Inmate)'
        homi_inm = [m_ma_dict[current], f_ma_dict[current]]
        current = 'Execution'
        execution = [m_ma_dict[current], f_ma_dict[current]]

        x_axis = np.arange(len(x_labels))

        fig, ax = plt.subplots()

        plt.bar(x_axis - 0.25, natural, 0.05, label='Natural')
        plt.bar(x_axis + 0.25, homi_just, 0.05, label='Homicide Justified: Law Enforcement')
        plt.bar(x_axis - 0.2, suicide, 0.05, label='Suicide')
        plt.bar(x_axis + 0.2, accident, 0.05, label='Accidental')
        plt.bar(x_axis - 0.15, pending, 0.05, label='Pending Investigation')
        plt.bar(x_axis + 0.15, homi_will, 0.05, label='Homicide Willful: Inmate')
        plt.bar(x_axis - 0.1, unknown, 0.05, label='Unknown')
        plt.bar(x_axis + 0.1, other, 0.05, label='Other')
        plt.bar(x_axis - 0.05, homi_law, 0.05, label='Homicide Willful: Law Enforcement')
        plt.bar(x_axis + 0.05, homi_inm, 0.05, label='Homicide Justified: Inmate')
        plt.bar(x_axis + 0.0, execution, 0.05, label='Execution')

        plt.xticks(x_axis, x_labels)
        plt.xlabel("Gender Labels")
        plt.ylabel("Number of Deaths")
        plt.title("Number of Deaths by Gender by Manner of Death")
        ax.xaxis_date()
        ax.autoscale(tight=True)
        plt.legend()
        fig.savefig(os.path.join(self.out_dir, f"gender_hist_manner.png"))

        # Generate the graphs
        #print("----------------------------------------------------------------------")
        #print(m_cs_dict)
        #print(f_cs_dict)

        current = 'Sentenced'
        sentenced = [m_cs_dict[current], f_cs_dict[current]]
        current = 'Process of Arrest'
        proc = [m_cs_dict[current], f_cs_dict[current]]
        current = 'Booked - Awaiting Trial'
        booked_aw = [m_cs_dict[current], f_cs_dict[current]]
        current = 'Booked - No Charges Filed'
        booked_no = [m_cs_dict[current], f_cs_dict[current]]
        current = 'Awaiting Booking'
        awaitb = [m_cs_dict[current], f_cs_dict[current]]
        current = 'Other'
        othe = [m_cs_dict[current], f_cs_dict[current]]
        current = 'In Transit'
        trans = [m_cs_dict[current], f_cs_dict[current]]
        current = 'Out to Court'
        court = [m_cs_dict[current], f_cs_dict[current]]

        x_axis = np.arange(len(x_labels))

        fig, ax = plt.subplots()

        plt.bar(x_axis - 0.2, sentenced, 0.05, label='Sentenced')
        plt.bar(x_axis + 0.15, proc, 0.05, label='Process of Arrest')
        plt.bar(x_axis - 0.15, booked_aw, 0.05, label='Booked - Awaiting Trial')
        plt.bar(x_axis + 0.1, booked_no, 0.05, label='Booked - No Charges Filed')
        plt.bar(x_axis - 0.1, awaitb, 0.05, label='Awaiting Booking')
        plt.bar(x_axis + 0.05, othe, 0.05, label='Other')
        plt.bar(x_axis - 0.05, trans, 0.05, label='In Transit')
        plt.bar(x_axis + 0.0, court, 0.05, label='Out to Court')

        plt.xticks(x_axis, x_labels)
        plt.xlabel("Gender Labels")
        plt.ylabel("Number of Deaths")
        plt.title("Number of Deaths by Gender by Custody Status")
        ax.xaxis_date()
        ax.autoscale(tight=True)
        plt.legend()
        fig.savefig(os.path.join(self.out_dir, f"gender_hist_custody.png"))


def main():
    """ Main """
    # Process directories
    dir_name = os.path.join(os.path.join(PATH, "out"), "DeathInCustody")
    out_dir = os.path.join(dir_name, TIME)
    os.makedirs(out_dir)

    # Initialize data class
    dataset = DataSet(os.path.join(os.path.join(PATH, "data"), "DeathInCustody_2005-2020_20210603.csv"), out_dir)
    dataset.process()

    # Core count for multi-proc
    core_count = round(multiprocessing.cpu_count() * .75)

    # Dependant counts
    #custody_counts = dataset.custody.value_counts()
    #print(custody_counts)
    #manner_counts = dataset.manner.value_counts()
    # Independent counts
    race_counts = dataset.race.value_counts()
    #age_counts = dataset.age.value_counts()
    #gender_counts = dataset.gender.value_counts()
    #print(gender_counts)

    # Step 3: Histograms
    # Generate Race Histograms
    #dataset.generate_race_hist()
    # Generate Age Histograms
    dataset.generate_age_hist()
    # Generate Gender Histograms
    dataset.generate_gender_hist()

    # Step 4: Two Stories, one variable
    x_labels = ['White', 'Black', 'Asian/Oceanic', 'Other', 'American Indian']
    x_data = [475, 312, 78, 21, 9]
    fig, ax = plt.subplots()
    ax.bar(x_labels, x_data)
    plt.xlabel("Racial Group of Prisoner")
    plt.ylabel("Number of Deaths")
    plt.title("Number of Justified Deaths by Law Enforcement")
    ax.autoscale(tight=True)
    fig.savefig(os.path.join(out_dir, f"stats_fair_graph.png"))

    x_labels = ['White', 'Hispanic', 'Black', 'Asian/Oceanic', 'Other', 'American Indian']
    white = 475/(race_counts['White'])
    hispa = 745/(race_counts['Hispanic'])
    black = 312/(race_counts['Black'])
    asian = 78/(race_counts['Asian/Oceanic'])
    other = 21/(race_counts['Other'])
    indian = 9/(race_counts['American Indian'])

    x_data = [white, hispa, black, asian, other, indian]
    fig, ax = plt.subplots()
    ax.bar(x_labels, x_data)
    plt.xlabel("Racial Group of Prisoner")
    plt.ylabel("Ratio of Total Deaths")
    plt.title("Ratio of Justified Deaths by Law Enforcement vs Total Deaths")
    ax.autoscale(tight=True)
    fig.savefig(os.path.join(out_dir, f"stats_real_graph.png"))

    # Step 5: Calculate Mean
    # Mean requires some fudging, so reverse severity
    x_data = [475, 745, 312, 78, 21, 9]
    total = x_data[0] + x_data[1] * 2 + x_data[2] * 3 + x_data[3] * 4 + x_data[4] * 5 + x_data[5] * 6
    count = x_data[0] + x_data[1] + x_data[2] + x_data[3] + x_data[4] + x_data[5]

    mean = total/count

    # Median is simpler
    sorted_race = dataset.race.sort_values()
    median = sorted_race[count/2]

    # Mode is easy by the count above
    mode = "Hispanic"

    print("Step 5:")
    print(f'Mean: {mean}')
    print(f'Median: {median}')
    print(f'Mode: {mode}')

    # Create Reduced data set and recalculate
    reduced_dataset = dataset.reduce()
    reduced_counts = reduced_dataset.race.value_counts()
    reduced_dataset.generate_race_hist()

    x_data = [234, 380, 175, 39, 8, 5]

    # Mean
    total = x_data[0] + x_data[1] * 2 + x_data[2] * 3 + x_data[3] * 4 + x_data[4] * 5 + x_data[5] * 6
    count = x_data[0] + x_data[1] + x_data[2] + x_data[3] + x_data[4] + x_data[5]
    mean = total/count

    # Median
    sorted_race = dataset.race.sort_values()
    median = sorted_race[int(count/2)]

    # Mode is easy by the count above
    mode = "Hispanic"

    print("Step 5: Reduced")
    print(f'Mean: {mean}')
    print(f'Median: {median}')
    print(f'Mode: {mode}')



if __name__ == "__main__":
    main()
