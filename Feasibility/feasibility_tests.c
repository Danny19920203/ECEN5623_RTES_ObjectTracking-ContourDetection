/**************************************************************************************************************************************************************************************************
 *
 *    File name: feasibility_tests.c
 *    date edited: May 1 2018
 *
 *    For a given set of computation values, time period and deadline, we need to determine if these values are feasible using
 *    the scheduling point test and the completion point test.
 *
 *    Credits: This code has been taken from Dr Sam Siewert's code
 * ************************************************************************************************************************************************************************************************/



#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>
#include<stdbool.h>
#include <math.h>
#define TRUE 1
#define FALSE 0

/*macros for service 1*/
#define SERVICE1_PERIOD ((float)215)
#define SERVICE1_DEADLINE ((float)185)
#define SERVICE1_EXECUTION_TIME ((float)177)

/*macros for service 2*/
#define SERVICE2_PERIOD ((float)215)
#define SERVICE2_DEADLINE ((float)10)
#define SERVICE2_EXECUTION_TIME ((float)7)

/*macros for service 3*/
#define SERVICE3_PERIOD ((float)215)
#define SERVICE3_DEADLINE ((float)20)
#define SERVICE3_EXECUTION_TIME ((float)18) 

/*period values for three services stored in an array*/
unsigned int RTES_PROJECT_period[] = {SERVICE1_PERIOD, SERVICE2_PERIOD, SERVICE3_PERIOD};

/*exacution time for three services stored in an array*/
unsigned int RTES_PROJECT_wcet[] = {SERVICE1_EXECUTION_TIME, SERVICE2_EXECUTION_TIME, SERVICE3_EXECUTION_TIME};

/*dealines for three services storerd in an array*/
unsigned int RTES_PROJECT_deadline[] = {SERVICE1_DEADLINE, SERVICE2_DEADLINE, SERVICE3_DEADLINE};

/*completion point test*/
int completion_time_feasibility(unsigned int no_of_services, unsigned int period[], unsigned int wcet[], unsigned int deadline[])
{
  int i, j;
  unsigned int an, anext;
  
  /*assume feasible until we find otherwise*/
  int set_feasible=TRUE;
   
  
  for (i=0; i < no_of_services; i++)
  {
       an=0; anext=0;
       
       for (j=0; j <= i; j++)
       {
           an+=wcet[j];
       }
       
       while(1)
       {
             anext=wcet[i];
	     
             for (j=0; j < i; j++)
                 anext += ceil(((double)an)/((double)period[j]))*wcet[j];
		 
             if (anext == an)
                break;
             else
                an=anext;
       }
       
       if (an > deadline[i])
       {
          set_feasible=FALSE;
       }
  }
  
  return set_feasible;
}

/*scheduling point test*/
int scheduling_point_feasibility(unsigned int no_of_services, unsigned int period[],unsigned int wcet[], unsigned int deadline[])
{
   int rc = TRUE, i, j, k, l, status, temp;

   /*iterate from highest to lowest priority*/
   for (i=0; i < no_of_services; i++) 
   {
      status=0;

      for (k=0; k<=i; k++) 
      {
          for (l=1; l <= (floor((double)period[i]/(double)period[k])); l++)
          {
               temp=0;

               for (j=0; j<=i; j++) temp += wcet[j] * ceil((double)l*(double)period[k]/(double)period[j]);

               if (temp <= (l*period[k]))
			   {
				   status=1;
				   break;
			   }
           }
           if (status) break;
      }
      if (!status) rc=FALSE;
   }
   return rc;
}

/*main function*/
int main(void)
{ 
    int i;
   
    unsigned int no_of_services;
    
    printf("-------------RTES FINAL PROJECT Completion Test Feasibility Example-----------------\n");
   
    printf("U=%4.2f (C1=177 ms,C2=7 ms,C3=18 ms; T1=215 ms, T2=215 ms, T3=215 ms; D1= 185 ms, D2= 10 ms, D3 = 20 ms): ",
		   ((SERVICE1_EXECUTION_TIME/SERVICE1_PERIOD) + (SERVICE2_EXECUTION_TIME/SERVICE2_PERIOD) + (SERVICE3_EXECUTION_TIME/SERVICE3_PERIOD)));

    no_of_services = sizeof(RTES_PROJECT_period)/sizeof(unsigned int);
    
    if(completion_time_feasibility(no_of_services, RTES_PROJECT_period, RTES_PROJECT_wcet, RTES_PROJECT_period) == TRUE)
        printf("FEASIBLE\n");
    else
        printf("INFEASIBLE\n");

    printf("\n\n");

    printf("--------------RTES FINAL PROJECT Scheduling Point Feasibility Example-----------------\n");
    
    printf("U=%4.2f (C1=177 ms, C2=7 ms, C3=18 ms; T1=215 ms, T2=215 ms, T3=215 ms; D1 = 185 ms, D2=10 ms,D3 = 20 ms): ",
		   (((SERVICE1_EXECUTION_TIME/SERVICE1_PERIOD) + (SERVICE2_EXECUTION_TIME/SERVICE2_PERIOD) + (SERVICE3_EXECUTION_TIME/SERVICE3_PERIOD))));
	
    no_of_services = sizeof(RTES_PROJECT_period)/sizeof(unsigned int);
    
    if(scheduling_point_feasibility(no_of_services, RTES_PROJECT_period, RTES_PROJECT_wcet, RTES_PROJECT_period) == TRUE)
        printf("FEASIBLE\n");
    else
        printf("INFEASIBLE\n");

}

