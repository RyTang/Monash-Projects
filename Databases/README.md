# Databases
This unit is based on learning the best practices to creating and designing a Database from scratch. Involves the usages of Oracle SQL to build a fully functional Database.
 
Things I have Learnt:
 1. The understanding of Database Paradigm
 1. Created a Conceptual model of a Database from a simulated client
 1. Normalisation Process
 1. Created a Logical Model from merging results of Normalisation and Conceptual
 1. Creation of Database and Insertion of Data
 1. Advanced Query Search of Databases

Below are the table of contents and projects that I've done in this unit. It will cover all the things mentioned in the list above and more. It'll show the process in which I've gone through to create a database from scratch and through clients requirements.

## __Disclaimer__:
The situation being depicted is not actually from a Real-Life Association. They are just modelled to project a scenario between a client and a designer as best as possible to train students.

## Table of Contents:
1. [Conceptual Model Design](#animal-doctors-conceptual-model)
1. [Normalisation Process Design](#animal-doctors-normalisation-process)
1. [Logical Model Design](#animal-doctors-logical-model)
1. [Animal Doctor Synopsis (If you're interested in the brief)](#animal-doctor-synopsis)


## Animal Doctors Conceptual Model
[File Preview](https://github.com/RyTang/Monash-Projects/blob/main/Databases/ad_conceptual.pdf)

This is a version of the conceptual model that I have created based on the situation given from Animal Doctors.

## Animal Doctors Normalisation Process
[File Preview](https://github.com/RyTang/Monash-Projects/blob/main/Databases/ad_normalisation.pdf)

Conducting the normalisation process based on administration forms used by Animal Doctors to make records. I apologise as I can't show any of the forms that was used in the normalisation process due to afflicted problems of copyright. However, the forms used by Animal Doctors were similar to other registration forms one would normally fill in. This is just meant to show the basic process of how I went from UNF to the 3NF which is then combined and prepared to be used in the logical model.

## Animal Doctors Logical Model
[File Preview](https://github.com/RyTang/Monash-Projects/blob/main/Databases/ad_logical.pdf)

The merging of the information gathered from both conceptual model and normalisation to create a logical model. The logical model was then created using Oracle.

## Animal Doctor Synopsis

Not all details will be stated here as it will be incredibly long and bothersome to read through. The synopsis is meant to give a brief understanding of the scenario behind Animal Doctors and some of the requirements they want in a database.

Animal Doctor is seeking a database which can be used to support the activities of a veterinary practice. Animal Doctors has a number of clinic distributed across the suburbs. For each clinic they want to be able to record the clinic's basic details and location. Afterwards, they want to be able to assign vets to one of the clinics as their home base. However, Specialists are allowed to rove around different clinics to offer their specialities. 

Pets will need to have details recorded alongside their owner's details. Each pet can only have one owner at any one time. Furthermore, whenever an owner brings a pet for a visit, Animal Doctors will like to have a record of the visit with details pertaining the inspection and pet. Furthermore, there might be a chance for follow-up visits which requires a record of the original visit. In each visitation, a vet might decide to prescribe medication for the animals and, thus, needs to be recorded by Animal Doctors for their usages and dosage given.
