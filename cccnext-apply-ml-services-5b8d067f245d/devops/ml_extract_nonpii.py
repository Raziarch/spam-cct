import os
from sys import argv
import pandas as pd
from datetime import datetime
from pytz import timezone


# Get current timestamp
def get_timestamp(tzinfo='US/Pacific'):
    tz = timezone(tzinfo)
    return datetime.now(tz).isoformat()


base_features = ["app_id", "email", "ccc_id", "status", "college_id", "term_id", "major_id", "intended_major",
                 "edu_goal", "highest_edu_level", "consent_indicator", "app_lang", "esignature", "ack_fin_aid",
                 "fin_aid_ref", "confirmation", "sup_page_code", "last_page", "city", "postalcode", "state",
                 "nonusaprovince", "country", "non_us_address", "perm_city", "perm_postalcode", "perm_state",
                 "perm_nonusaprovince", "perm_country", "address_same", "enroll_status", "hs_edu_level", "hs_comp_date",
                 "higher_edu_level", "higher_comp_date", "hs_not_attended", "cahs_graduated", "cahs_3year", "hs_name",
                 "hs_city", "hs_state", "hs_country", "hs_cds", "hs_ceeb", "hs_not_listed", "home_schooled",
                 "college_count", "hs_attendance", "coenroll_confirm", "gender", "pg_rel", "pg1_edu", "pg2_edu",
                 "pg_edu_mis", "under19_ind", "dependent_status", "race_ethnic", "hispanic", "race_group", "salt",
                 "citizenship_status", "visa_type", "no_documents", "military_status", "military_discharge_date",
                 "military_home_state", "military_home_country", "military_ca_stationed", "military_legal_residence",
                 "ca_res_2_years", "ca_date_current", "ca_not_arrived", "ca_college_employee", "ca_school_employee",
                 "ca_seasonal_ag", "ca_foster_youth", "ca_outside_tax", "ca_outside_tax_year", "ca_outside_voted",
                 "ca_outside_voted_year", "ca_outside_college", "ca_outside_college_year", "ca_outside_lawsuit",
                 "ca_outside_lawsuit_year", "res_status", "res_status_change", "res_prev_date", "adm_ineligible",
                 "elig_ab540", "res_area_a", "res_area_b", "res_area_c", "res_area_d", "experience", "recommend",
                 "comments", "comfortable_english", "financial_assistance", "tanf_ssi_ga", "foster_youths",
                 "athletic_intercollegiate", "athletic_intramural", "athletic_not_interested", "academic_counseling",
                 "basic_skills", "calworks", "career_planning", "child_care", "counseling_personal", "dsps", "eops",
                 "esl", "health_services", "housing_info", "employment_assistance", "online_classes", "reentry_program",
                 "scholarship_info", "student_government", "testing_assessment", "transfer_info", "tutoring_services",
                 "veterans_services", "integrity_fg_01", "integrity_fg_02", "integrity_fg_03", "integrity_fg_04",
                 "integrity_fg_11", "integrity_fg_47", "integrity_fg_48", "integrity_fg_49", "integrity_fg_50",
                 "integrity_fg_51", "integrity_fg_52", "integrity_fg_53", "integrity_fg_54", "integrity_fg_55",
                 "integrity_fg_56", "integrity_fg_57", "integrity_fg_58", "integrity_fg_59", "integrity_fg_60",
                 "integrity_fg_61", "integrity_fg_62", "integrity_fg_63", "integrity_fg_70", "integrity_fg_80",
                 "col1_ceeb", "col1_cds", "col1_not_listed", "col1_name", "col1_city", "col1_state", "col1_country",
                 "col1_start_date", "col1_end_date", "col1_degree_date", "col1_degree_obtained", "col2_ceeb",
                 "col2_cds", "col2_not_listed", "col2_name", "col2_city", "col2_state", "col2_country",
                 "col2_start_date", "col2_end_date", "col2_degree_date", "col2_degree_obtained", "col3_ceeb",
                 "col3_cds", "col3_not_listed", "col3_name", "col3_city", "col3_state", "col3_country",
                 "col3_start_date", "col3_end_date", "col3_degree_date", "col3_degree_obtained", "col4_ceeb",
                 "col4_cds", "col4_not_listed", "col4_name", "col4_city", "col4_state", "col4_country",
                 "col4_start_date", "col4_end_date", "col4_degree_date", "col4_degree_obtained", "college_name",
                 "district_name", "term_code", "term_description", "major_code", "major_description", "tstmp_submit",
                 "tstmp_create", "tstmp_update", "foster_youth_status", "foster_youth_preference", "foster_youth_mis",
                 "foster_youth_priority", "tstmp_download", "address_validation", "zip4", "perm_address_validation",
                 "perm_zip4", "discharge_type", "college_expelled_summary", "col1_expelled_status",
                 "col2_expelled_status", "col3_expelled_status", "col4_expelled_status", "integrity_flags", "rdd",
                 "ssn_type", "military_stationed_ca_ed", "integrity_fg_65", "integrity_fg_64", "ip_address",
                 "campaign1", "campaign2", "campaign3", "ssn_exception", "integrity_fg_71", "ssn_no",
                 "completed_eleventh_grade", "grade_point_average", "highest_english_course", "highest_english_grade",
                 "highest_math_course_taken", "highest_math_taken_grade", "highest_math_course_passed",
                 "highest_math_passed_grade", "integrity_fg_30", "hs_cds_full", "col1_cds_full", "col2_cds_full",
                 "col3_cds_full", "col4_cds_full", "no_perm_address_homeless", "no_mailing_address_homeless",
                 "ccgi_token", "homeless_youth", "integrity_fg_40", "term_start", "term_end", "cip_code",
                 "major_category", "confirmed_fraud", "fraud_status", "fraud_score"]

tstmp = get_timestamp(tzinfo='US/Pacific')

infile = argv[1]
outfile = os.path.splitext(infile)[0] + '_' + tstmp + '_nonpii.csv'

df = pd.read_csv(infile, dtype=object, usecols=base_features)
df['email'] = df['email'].apply(lambda x: x.split('@')[1])
df.to_csv(outfile, index=False)
