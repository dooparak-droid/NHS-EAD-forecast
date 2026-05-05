# =============================================================================
# AGGREGATION MAP: All metrics classified by daily aggregation method
# =============================================================================
# Methods:
#   "mean" = snapshot/stock metrics, rates, percentages, time averages
#   "max"  = cumulative-since-midnight counters (take end-of-day value)
#   "sum"  = rolling last-hour flow metrics (sum hourly buckets for daily total)
#   "last" = yesterday's data metrics (take single reported value, then lag by 1 day)
#   "asis" = already daily (uec_daily source); just take the single value
# =============================================================================

library(tibble)

aggregation_map <- tribble(
  ~metric_name,                                                                                          ~agg_method,

  # ── A&E / ED OPERATIONAL METRICS (snapshot / stock) ─────────────────────────
  "Patients in A&E",                                                                                     "mean",
  "No. of DTAs",                                                                                         "mean",
  "No. of DTAs (> 8hrs)",                                                                                "mean",
  "Resuscitation Capacity",                                                                              "mean",
  "Resuscitation Capacity at 1000",                                                                      "mean",
  "Minors patient count",                                                                                "mean",
  "Majors patient count",                                                                                "mean",
  "Cohorting & Reverse Queue",                                                                           "mean",
  "Ambulance Queue",                                                                                     "mean",
  "Number of Medical Outliers",                                                                          "mean",
  "Number of Surgical Outliers",                                                                         "mean",
  "Number of Cardiac Outliers",                                                                          "mean",
  "Number of Outliers (excluding paediatrics) at 1000",                                                  "mean",
  "Number of GP expected diverts to ED",                                                                 "mean",

  # ── A&E RATES / TIME AVERAGES ────────────────────────────────────────────────
  "4hr Breach Performance",                                                                              "mean",
  "Average Time to Triage",                                                                              "mean",
  "Average Time to Assessment",                                                                          "mean",

  # ── A&E CUMULATIVE SINCE MIDNIGHT → take MAX (end-of-day value) ─────────────
  "Total Breaches Since Midnight",                                                                       "max",
  "5. Median time to treatment since midnight",                                                          "max",

  # ── A&E LAST HOUR FLOWS → SUM hourly buckets for daily total ────────────────
  "New Arrivals in Last Hour",                                                                           "sum",
  "A&E Discharges in Last Hour",                                                                         "sum",

  # ── AMBULANCE / HANDOVER METRICS ─────────────────────────────────────────────
  # Last-hour flows: sum for daily total
  "Ambulance Handovers 15mins (Last Hour)",                                                              "sum",
  "Ambulance Handovers 30 mins (Last Hour)",                                                             "sum",
  "Ambulance Handovers 60 mins (Last Hour)",                                                             "sum",
  "Handover to Clear 15mins (Last Hour)",                                                                "sum",
  "Handover to Clear 30mins (Last Hour)",                                                                "sum",
  "Ambulances Conveyed to Hospital (Last Hour)",                                                         "sum",

  # Since-midnight cumulative: take MAX
  "Ambulance Handovers 15mins (Since Midnight)",                                                         "max",
  "Ambulance Handovers 30mins (Since Midnight)",                                                         "max",
  "Ambulance Handovers 60mins (Since Midnight)",                                                         "max",
  "Handover to Clear 15mins (Since Midnight)",                                                           "max",
  "Handover to Clear 30mins (Since Midnight)",                                                           "max",
  "Handover to Clear 60mins (Since Midnight)",                                                           "max",
  "Ambulances Conveyed to Hospital (Since Midnight)",                                                    "max",
  "Handover Time Lost Since Midnight (hh:mm)",                                                           "max",

  # Snapshot: current count at any given time
  "Ambulances En Route to Hospital",                                                                     "mean",
  "Ambulances En Route to Hospitals",                                                                    "mean",

  # ── 999 CALL STACK ────────────────────────────────────────────────────────────
  "Number of Active Calls on the 999 Call Stack",                                                        "mean",
  "Number of Waiting Calls on the 999 Call Stack",                                                       "mean",

  # ── AMBULANCE CATEGORY RESPONSE TIMES (cumulative since midnight) ────────────
  "Category 1 - BNSSG Mean Response (Since Midnight)",                                                   "max",
  "Category 2 - BNSSG Mean Response (Since Midnight)",                                                   "max",
  "Category 3 - BNSSG Mean Response (Since Midnight)",                                                   "max",
  "Category 4 - BNSSG Mean Response (Since Midnight)",                                                   "max",
  # 90th percentile - last hour snapshot
  "Category 1 - BNSSG 90th Response (Last Hour)",                                                        "mean",
  "Category 2 - BNSSG 90th Response (Last Hour)",                                                        "mean",
  "Category 3 - BNSSG 90th Response (Last Hour)",                                                        "mean",
  "Category 4 - BNSSG 90th Response (Last Hour)",                                                        "mean",

  # Proportions (rates): mean
  "Proportion of Calls Since Midnight that Received a See Convey",                                       "mean",
  "Proportion of Calls Since Midnight that Received a See Treat",                                        "mean",
  "Proportion of Calls Since Midnight that Received a Hear Treat",                                       "mean",

  # ── SEVERNSIDE / NHS 111 / IUC ───────────────────────────────────────────────
  # Cumulative since midnight / daily reported totals
  "(Severnside) Calls Received",                                                                         "max",
  "(Severnside) Calls Answered",                                                                         "max",
  "(Severnside) HCP Calls Answered",                                                                     "max",
  "(Severnside) Calls Answered Within 60 Seconds",                                                       "max",
  "(Severnside) Calls Triaged",                                                                          "max",
  "(Severnside) Referred to 999 - C1 Ambulances",                                                        "max",
  "(Severnside) Referred to 999 - C2 Ambulances",                                                        "max",
  "(Severnside) Referred to 999 - C3 Ambulances",                                                        "max",
  "(Severnside) Referred to 999 - C4 Ambulances",                                                        "max",
  "(Severnside) Referred to ED",                                                                         "max",
  "(Severnside) CAS Referred to ED",                                                                     "max",

  # IUC snapshots / longest waits
  "Number of Cases on IUC CAS Queue",                                                                    "mean",
  "Number of Home Visits",                                                                               "mean",
  "Number of Treatment Centre Appointments",                                                             "mean",
  "IUC Advice Longest Wait (HCP Call Back)",                                                             "mean",
  "IUC Advice Longest Wait (1hr Call Back)",                                                             "mean",
  "IUC Advice Longest Wait (2hr Call Back)",                                                             "mean",
  "IUC Advice Longest Wait (6hr Call Back)",                                                             "mean",
  "Clinical Escalation Level (BrisDoc)",                                                                 "mean",

  # ── SWASFT ────────────────────────────────────────────────────────────────────
  "(SWASFT) Number of HCP Incidents",                                                                    "max",
  "(SWASFT) Number of NHS 111 Incidents",                                                                "max",

  # ── CALL CENTRE METRICS (15-min rolling windows) ─────────────────────────────
  # Last 15 mins: sum gives daily total for counts
  "Calls Offered in Last 15 mins",                                                                       "sum",
  "Calls Answered in Last 15 mins",                                                                      "sum",
  "Calls Abandoned Before Threshold in Last 15 mins",                                                    "sum",
  "Calls Abandoned After Threshold in Last 15 mins",                                                     "sum",
  "Answered in 60 seconds in Last 15 mins",                                                              "sum",
  # Rates and times: mean
  "Average Speed to Answer in Last 15 mins",                                                             "mean",
  "Average Handling Time in Last 15 mins",                                                               "mean",
  "Longest Wait in Last 15 mins",                                                                        "mean",
  "Calls Abandoned After Threshold in Last 15 mins (%)",                                                 "mean",
  "Answered in 60 seconds in Last 15 mins (%)",                                                          "mean",

  # Since midnight cumulative call centre: take MAX
  "Calls Offered Since Midnight that Received",                                                          "max",
  "Calls Answered Since Midnight that Received",                                                         "max",
  "Calls Abandoned Before Threshold Since Midnight that Received",                                       "max",
  "Calls Abandoned After Threshold Since Midnight that Received",                                        "max",
  "Answered in 60 seconds Since Midnight that Received",                                                 "max",
  # Rates and times for since midnight: mean
  "Average Speed to Answer Since Midnight that Received",                                                "mean",
  "Average Handling Time Since Midnight that Received",                                                  "mean",
  "Longest Wait Since Midnight that Received",                                                           "mean",
  "Calls Abandoned After Threshold Since Midnight that Received (%)",                                    "mean",
  "Answered in 60 seconds Since Midnight that Received (%)",                                             "mean",

  # ── OPEL / ESCALATION ────────────────────────────────────────────────────────
  # Ordinal snapshot: take mean (treat as continuous for modelling)
  "OPEL",                                                                                                "mean",
  "Automated OPEL",                                                                                      "mean",
  "Aggregated NHSE OPEL Score",                                                                          "mean",
  "DOS Status at 10am",                                                                                  "mean",

  # ── BED / CAPACITY METRICS ───────────────────────────────────────────────────
  # Snapshots at fixed times of day (uec_daily): as-is (single daily value)
  "Number of critical care beds available at 8am (today)",                                               "asis",
  "Number of empty beds on assessment units 8am (today)",                                                "asis",
  "Number of empty beds on assessment units 8am (today) - Medicine",                                     "asis",
  "Number of empty beds on assessment units 8am (today) - Surgery",                                      "asis",
  "Deficit of Discharges at 1100 - With Definite Discharges Known",                                      "asis",
  "Deficit of Discharges at 1100 - With Definite and Potential Discharges Identified",                   "asis",
  "Escalation beds open",                                                                                "asis",
  "G&A Bed occupancy",                                                                                   "asis",
  "G&A beds, core stock open",                                                                           "asis",
  "Beds occupied by long-stay patients (21+ days)",                                                      "asis",
  "Adult critical care beds occupied",                                                                   "asis",
  "Critical Care beds - Adult critical care beds occupied",                                               "asis",
  "Critical Care beds - Paediatric intensive care beds occupied",                                         "asis",
  "Critical Care beds - Neonatal intensive care cots occupied",                                           "asis",
  "Of neonatal critical care beds occupied, how many are level 3?",                                       "asis",
  "A&E attends - paediatrics",                                                                           "asis",
  "Number of Discharges",                                                                                "asis",
  "Number of Admissions",                                                                                "asis",
  "Bed Occupancy Adult",                                                                                 "asis",
  "Bed Occupancy PICU",                                                                                  "asis",

  # NHSE daily metrics
  "2. ED all-type 4-hour performance",                                                                   "asis",
  "3. ED all-type attendance variation",                                                                  "asis",
  "4. Majors and resuscitation occupancy (adult)",                                                        "asis",
  "6. % of patients spending >12 hours in ED",                                                           "asis",
  "8. % of open beds that are escalation beds",                                                           "asis",
  "9. % of beds occupied by patients with NCtR",                                                         "asis",

  # ── DtA PATHWAY METRICS (discharge-to-assess) ────────────────────────────────
  # Waiting lists / bed counts: snapshot, mean
  "DtA Community P2 Bed Occupied",                                                                       "mean",
  "DtA P1 BOOKED AND UN-BOOKED Waiting for capacity, medically fit and NOT in an acute",                 "mean",
  "DtA P1 BOOKED AND UN-BOOKED Waiting for capacity, medically fit and ready to leave ACUTE",            "mean",
  "DtA P1 BOOKED Waiting for capacity, medically fit and NOT in an acute",                               "mean",
  "DtA P1 BOOKED Waiting for capacity, medically fit and ready to leave ACUTE",                          "mean",
  "DtA P1 UNBOOKED Waiting for capacity, medically fit and NOT in an acute",                             "mean",
  "DtA P1 UNBOOKED Waiting for capacity, medically fit and ready to leave acute",                        "mean",
  "DtA P1 TOTAL (BOOKED AND UNBOOKED) Waiting for capacity and medically fit, and in any setting (acute and non-acute)", "mean",
  "DtA P2 Beds Occupied",                                                                                "mean",
  "DtA P2 BOOKED Waiting for capacity, medically fit and ready to leave ACUTE",                          "mean",
  "DtA P2 UNBOOKED Waiting for capacity, medically fit and ready to leave acute",                        "mean",
  "DtA P2 TOTAL BOOKED AND UN-BOOKED Waiting for capacity, medically fit and ready to leave ACUTE",      "mean",
  "DtA P2 referrals received",                                                                           "mean",
  "DtA P3 BOOKED AND UN-BOOKED Waiting for capacity, medically fit and ready to leave ACUTE",            "mean",
  "DtA P3 BOOKED Waiting for capacity, medically fit and ready to leave ACUTE",                          "mean",
  "DtA P3 UNBOOKED Waiting for capacity, medically fit and ready to leave acute",                        "mean",
  "DtA P3 TOTAL (BOOKED AND UNBOOKED) Waiting for capacity and medically fit and in any setting (acute and transitional bed)", "mean",
  "DtA P3 beds occupied",                                                                                "mean",
  "DtA P3 Beds Occupied",                                                                                "mean",
  "DtA SBRU P2 rehabilitation beds occupied",                                                            "mean",
  "DtA SBRU P2 stroke beds occupied",                                                                    "mean",
  "DtA P1 Referrals Received",                                                                           "mean",
  "DtA P1 slots booked",                                                                                 "mean",
  "DtA P1 Slots Booked",                                                                                 "mean",
  "DtA P2 Referrals Received",                                                                           "mean",
  "No of referrals received for DtA P3",                                                                 "mean",
  "Closed TOC Docs to all pathways",                                                                     "mean",
  "Delayed TOC Docs to all Pathways",                                                                    "mean",

  # ── NCtR METRICS (no criteria to reside) ─────────────────────────────────────
  # Snapshot counts: mean
  "BRI NCtR Beddays",                                                                                    "mean",
  "BRI NCtR Patients",                                                                                   "mean",
  "BRI P1 NCtR Beddays",                                                                                 "mean",
  "BRI P1 NCtR Patients",                                                                                "mean",
  "BRI P2 NCtR Beddays",                                                                                 "mean",
  "BRI P2 NCtR Patients",                                                                                "mean",
  "BRI P3 NCtR Beddays",                                                                                 "mean",
  "BRI P3 NCtR Patients",                                                                                "mean",
  "BRI NCtR >= 65 years old",                                                                            "mean",
  "NBT NCtR Beddays",                                                                                    "mean",
  "NBT NCtR Patients",                                                                                   "mean",
  "NBT P1 NCtR Beddays",                                                                                 "mean",
  "NBT P1 NCtR Patients",                                                                                "mean",
  "NBT P2 NCtR Beddays",                                                                                 "mean",
  "NBT P2 NCtR Patients",                                                                                "mean",
  "NBT P3 NCtR Beddays",                                                                                 "mean",
  "NBT P3 NCtR Patients",                                                                                "mean",
  "NBT NCtR >= 65 years old",                                                                            "mean",
  "WGH NCtR Beddays",                                                                                    "mean",
  "WGH NCtR Patients",                                                                                   "mean",
  "WGH P1 NCtR Beddays",                                                                                 "mean",
  "WGH P1 NCtR Patients",                                                                                "mean",
  "WGH P2 NCtR Beddays",                                                                                 "mean",
  "WGH P2 NCtR Patients",                                                                                "mean",
  "WGH P3 NCtR Beddays",                                                                                 "mean",
  "WGH P3 NCtR Patients",                                                                                "mean",
  "WGH NCtR >= 65 years old",                                                                            "mean",
  "P1 NCtR",                                                                                             "mean",
  "P2 NCtR",                                                                                             "mean",

  # ── PATHWAY SLOT / BED AVAILABILITY ─────────────────────────────────────────
  "P1 Slots Booked Today",                                                                               "mean",
  "P1 (Available) Slots Not Utilised",                                                                   "mean",
  "P2 Beds Occupied",                                                                                    "mean",
  "P2 (Available) Beds Not Utilised",                                                                    "mean",
  "P3 Beds Occupied",                                                                                    "mean",
  "P3 (Available) Beds Not Utilised",                                                                    "mean",
  "P1 Acute Waiting List (Booked + Unbooked)",                                                           "mean",
  "P2 Acute Waiting List (Booked + Unbooked)",                                                           "mean",
  "P3 Acute Waiting List (Booked + Unbooked)",                                                           "mean",
  "P2 Patients Predicted for Admission Today",                                                           "mean",
  "P1 Caseload",                                                                                         "mean",
  "Intensive Caseload",                                                                                  "mean",
  "Capacity (Available Slots Today)",                                                                     "mean",
  "Mason Section 136 Beds Available (Capacity 4)",                                                        "mean",

  # ── YESTERDAY'S DATA → take as-is, then lag by 1 day in pipeline ─────────────
  "Patients Admitted to DtA P2 - YESTERDAY'S DATA",                                                     "last",
  "Patients Admitted to DtA P3 - YESTERDAY'S DATA",                                                     "last",
  "P1 Slots Used Yesterday",                                                                             "last",
  "P1 Referrals Yesterday",                                                                              "last",
  "P2 Referrals Yesterday",                                                                              "last",
  "P2 Admissions Yesterday",                                                                             "last",
  "P3 Referrals Yesterday",                                                                              "last",
  "P3 Admissions Yesterday",                                                                             "last",
  "Referrals Received Yesterday",                                                                        "last",
  "WGH P0 Discharges",                                                                                   "last",
  "WGH Complex Discharges",                                                                              "last",
  "BRI P0 Discharges",                                                                                   "last",
  "BRI Complex Discharges",                                                                              "last",
  "NBT P0 Discharges",                                                                                   "last",
  "NBT Complex Discharges",                                                                              "last",

  # ── SIRONA STAFFING ───────────────────────────────────────────────────────────
  # Percentage staffing reduction: mean
  "Sirona BNSSG % Staffing Reduction",                                                                   "mean",
  "Sirona ICE Bristol Locality % Staffing Reduction",                                                    "mean",
  "Sirona N&W Bristol Locality % Staffing Reduction",                                                    "mean",
  "Sirona SASS % Staffing Reduction",                                                                    "mean",
  "Sirona South Bristol Locality % Staffing Reduction",                                                  "mean",
  "Sirona South Glos Locality % Staffing Reduction",                                                     "mean",
  "Sirona Weston & Worle Locality % Staffing Reduction",                                                 "mean",
  "Sirona Woodspring Locality % Staffing Reduction",                                                     "mean",
  "Unallocated Red Nursing Schedules",                                                                   "mean",
  "Red Nursing schedules",                                                                               "mean",
  "Unallocated Red Nursing Schedules (%)",                                                               "mean",

  # ── OTHER ─────────────────────────────────────────────────────────────────────
  "Urgent Referrals",                                                                                    "mean",
  "OOT Internal Placements",                                                                             "mean",
)

# =============================================================================
# VALIDATION: Check all metrics in metadata are covered
# =============================================================================
# Run this to verify no metric is missing from the map:
#
# metadata <- data.table::fread("data/metric_metadata.csv")
# missing_from_map <- setdiff(unique(metadata$metric_name), aggregation_map$metric_name)
# if (length(missing_from_map) == 0) {
#   message("All metrics covered.")
# } else {
#   warning("These metrics are NOT in the aggregation map:\n",
#           paste(missing_from_map, collapse = "\n"))
# }
