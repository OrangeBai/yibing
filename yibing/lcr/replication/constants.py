# Inflow
# inflow = `loan_1y` * 0.5/12 + 1 * `lcr_deri_inflow`

# RC-C Memo
LOANS_AND_LEASES_1Y = "RCONA247"    # FFIEC 031, RC-C Part 1 Memo, 1.c
REPRICING_DATA_3M = "RCONA564"      # FFIEC 031, RC-C Part 1 Memo, 2.a.(1)
REPRICING_DATA_1Y = "RCONA565"      # FFIEC 031, RC-C Part 1 Memo, 2.a.(2)
NODE_LOAN_1Y = (LOANS_AND_LEASES_1Y, REPRICING_DATA_3M, REPRICING_DATA_1Y)
# LOAN_1Y = LOANS_AND_LEASES_1Y + REPRICING_DATA_3M + REPRICING_DATA_1Y

# Derivatives
# Notice that in the following, RCFD is from 031, RCON is from 041,
# Use list to compute the max.

# Derivatives with postive pair value held for purpose other than trading
# TODO Some tweaks here? 
DERI_RECV = ["RCONC010", "RCFDC010"]    # RC-F 6.c
CDS_POS = (['RCONC219', 'RCFDC219'], ['RCONC221', 'RCFDC221']) # RC-L 7.b

# CDS_POS = CDS_PURC_POS_FAIR_VALUE + CDS_SOLD_POS_FAIR_VALUE


DERI_INT_RATE_FAIR_VALUE_POS_HOLD_FOR_TRADING = ['RCON8733', "RCFD8733"] # RC-L 15.a.(1)
DERI_FOR_EXCH_FAIR_VALUE_POS_HOLD_FOR_TRADING = ['RCON8734', "RCFD8734"] # RC-L 15.a.(1)
DERI_EQU_DERI_FAIR_VALUE_POS_HOLD_FOR_TRADING = ['RCON8735', "RCFD8735"] # RC-L 15.a.(1)
DERI_COM_OTHE_FAIR_VALUE_POS_HOLD_FOR_TRADING = ['RCON8736', "RCFD8736"] # RC-L 15.a.(1)

DERI_INT_RATE_FAIR_VALUE_POS_HOLD_NOT_TRADING = ['RCON8741', "RCFD8741"] # RC-L 15.b.(1)
DERI_FOR_EXCH_FAIR_VALUE_POS_HOLD_NOT_TRADING = ['RCON8742', "RCFD8742"] # RC-L 15.b.(1)
DERI_EQU_DERI_FAIR_VALUE_POS_HOLD_NOT_TRADING = ['RCON8743', "RCFD8743"] # RC-L 15.b.(1)
DERI_COM_OTHE_FAIR_VALUE_POS_HOLD_NOT_TRADING = ['RCON8744', "RCFD8744"] # RC-L 15.b.(1)

# lcr_deri_inflow=   #  1.00*deri_recv+   #  1.00*cds_pos+  #  1.00*deri_int_pos_trd+  #  1.00*deri_fx_pos_trd+  #  1.00*deri_eq_pos_trd+  #  1.00*deri_cmd_pos_trd+  #  1.00*deri_int_pos_ntrd+  #  1.00*deri_fx_pos_ntrd+  #  1.00*deri_eq_pos_ntrd+  #  1.00*deri_cmd_pos_ntrd
NODE_DERI_INFLOW = (
    DERI_RECV,  # deri recv
    CDS_POS, 
    DERI_INT_RATE_FAIR_VALUE_POS_HOLD_FOR_TRADING,
    DERI_FOR_EXCH_FAIR_VALUE_POS_HOLD_FOR_TRADING,
    DERI_EQU_DERI_FAIR_VALUE_POS_HOLD_FOR_TRADING,
    DERI_COM_OTHE_FAIR_VALUE_POS_HOLD_FOR_TRADING,
    DERI_INT_RATE_FAIR_VALUE_POS_HOLD_NOT_TRADING,
    DERI_FOR_EXCH_FAIR_VALUE_POS_HOLD_NOT_TRADING,
    DERI_EQU_DERI_FAIR_VALUE_POS_HOLD_NOT_TRADING,
    DERI_COM_OTHE_FAIR_VALUE_POS_HOLD_NOT_TRADING,
)

# Outflow


# Part 1 Stable retail deposit
# stable retail deposit = 0.05*trans_deposit_ipb*(1-uninsured_ratio)*s1_retail+   0.05*tdlt100k_3mp*s1_1m_q*(1-uninsured_ratio)+   0.05*saving_deposit*(1-uninsured_ratio)*s1_retail
S1_1M_Q = 1/3
S1_1M_Y= 1.0/12.0

S1_RETAIL = 0.5
TRANSACTION_IND_COR = "RCONB549"    # RC-E, 1, Individual, Partnership, and Corporation
TRANSACTION_DOM_GOV = "RCON2202"    # RC-E, 2, US Government
TRANSACTION_SUB_GOV = "RCON2203"    # RC-E, 3, States and political subdivisions.
TRANSACTION_DOM_BNK = "RCONB551"    # RC-4, 4, US Banks.
TRANSACTION_FOR_BNK = "RCON2213"    # RC-E, 5, Banks in foreign countries.
TRANSACTION_FOR_GOV = "RCON2216"

SAVING_DEPOSIT = ("RCON6810", "RCON0352")
DEPOSIT_FOREIGN_OFC = "RCFN2200"    # RC, Liabilities, 13.b. Deposits in foreign offices.

# TODO Verify this.
TIME_DEPOSIT_LESS_250K_LESS_3M = "RCONHK07" # RC-E Part 1, Memo 3.a.(1). Notice that the original report using 100K from 034, but no longer available in 031/041
TIME_DEPOSIT_LESS_250K_LESS_1Y = "RCONHK11"

DEPOSIT = ('RCON2200', 'RCFN2200')        # RC-E Part 1 7
UNINSURED_DEPOSIT = 'RCON5597'  # RC-Q, Memo 2, estimated.
# uninsured_ratio = UNINSURED_DEPOSIT / DEPOSIT

# Part 2 Less stable retail deposit
# Less stable retail deposit = 0.10*trans_deposit_ipb*uninsured_ratio*s1_retail+  /// 0.10*tdlt100k_3mp*s1_1m_q*uninsured_ratio+  /// 0.10*saving_deposit*uninsured_ratio*s1_retail+ ///

# Part 3 stable operational deposit
# stable operational deposit = 0.05*trans_deposit_ipb*(1-uninsured_ratio)*(1-s1_retail)+   0.05*trans_deposit_dgov*(1-uninsured_ratio)+   0.05*trans_deposit_sgov*(1-uninsured_ratio)+   0.05*trans_deposit_dbank*(1-uninsured_ratio)+   0.05*trans_deposit_fbank*(1-uninsured_ratio)+   0.05*trans_deposit_fgov*(1-uninsured_ratio)+   0.05*fn_deposit*s1_1m_y*(1-uninsured_ratio)

# Part 4 less stable operational deposit
# less stable operational deposit = 0.25*trans_deposit_ipb*uninsured_ratio*(1-s1_retail)+   0.25*trans_deposit_dgov*uninsured_ratio+   0.25*trans_deposit_sgov*uninsured_ratio+   0.25*trans_deposit_dbank*uninsured_ratio+   0.25*trans_deposit_fbank*uninsured_ratio+   0.25*trans_deposit_fgov*uninsured_ratio+   0.25*fn_deposit*s1_1m_y*uninsured_ratio

# Part 5 stable non-financial corporate, sovereigns, central banks
# SAVING_DEPOSIT = "RCON0352"
# stable non-financial corporate, sovereigns, central banks, pse = 0.75*saving_deposit*(1-uninsured_ratio)*(1-s1_retail)+  0.75*tdge100k_3mp*(1-uninsured_ratio)*s1_1m_q+  

# Part 6 less stable non-financial corporate, sovereigns, central banks, pse
# less stable non-financial corporate, sovereigns, central banks, pse = 0.75*saving_deposit*uninsured_ratio*(1-s1_retail) + 0.75*tdge100k_3mp*uninsured_ratio*s1_1m_q

# Part 7 secured lending backed by level 2 asset
# secured lending backed by level 2 asset = 0.15*rw00_sec_lent+  0.15*repo_sec+  0.15*rw20_sec_lent+  

RW00_SEC_LENT = ['RCFDS517', 'RCONS517']   # RC-R, Part 2, 16. The original code incorporates the securities lent (RCONB665), while this is removed after 2014 and included in code RCONS517.
REPO_SEC = ['RCFDB995', 'RCONB995']         # RC, Liabilities, 14.b. Federal funds purchased and securities purchased under agreements to resell, Securities sold under agreements to repurchase Same field from FFIEC 034, RCON0279, ignored here.
RW20_SEC_LENT_031 = {'RCONS518', 'RCONS519', 'RCONS520'}
RW20_SEC_LENT_041 = {'RCFDS518', 'RCFDS519', 'RCFDS520'}
# RW20_SEC_LENT = max(RW20_SEC_LENT_031, RW20_SEC_LENT_041)

# Part 8 All other secured funding transactions
# all other secured funding transactions = 1.00*repo_ffund + 1.00*rw50_sec_lent+ 1.00*rw100_sec_lent
REPO_FFUND = ["RCONB989", "RCFDB989"]       # RC, Assets, 3.b. Federal funds sold and securities purchased under agreements to resell,  securities purchased under agreements to resell. 
RW50_SEC_LENT = ["RCONS521", "RCFDS521"]    # RC-R, 16. Repo-style transactions 50% risk weight category. # Notice that here we are missing RCONB667.
RW100_SEC_LENT = ["RCONSD522", "RCFDS522"]  # RC-R, 16, 100% Risk weight category.

# Part 9 Other liabilities.
# Other liabilities = 1.00*other_liab+ 1.00*trading_liab+ 1.00*other_bm_1y*s1_1m_y
OTHER_LIABILITIES = ["RCFD2930", "RCON2930"]    # RC-G, 5. Total other liabilities.
TRAINDG_LIABILITIES = ["RCFD3548", "RCON3548"]  # RC, 15, Trading Liabilities.
# other bm 1y = 0.25 * fhlb_advance_1y 
FED_HOME_LOAN_BANK_ADVANCE = ["RCFDF055", "RCONF055", "RCFD2651", "RCON2651"]   # RC-M, Federal Home Loan Bank advances, one year or less

OTHER_BORROWING = ["RCFDF060", "RCONF060", "RCFDB571", "RCONB571"] # RC-M, 5.b.(1).(a)/5.b.(2)

# Part 9 Derivative outflow
# lcr_deri_outflow =   1.00*cds_neg+  1.00*deri_pay+  1.00*deri_int_neg_trd+  1.00*deri_fx_neg_trd+  1.00*deri_eq_neg_trd+  1.00*deri_cmd_neg_trd+  1.00*deri_int_neg_ntrd+  1.00*deri_fx_neg_ntrd+  1.00*deri_eq_neg_ntrd+  1.00*deri_cmd_neg_ntrd 
CDS_SOLD_NEG_FAIR_VALUE = ['RCONC220', 'RCFDC220']  # RC-L 7.b.(2)
CDS_PURC_NEG_FAIR_VALUE = ['RCONC222', 'RCONC222']  # RC-L 7.b.(2)
# CDS_NEG = CDS_PURC_NEG_FAIR_VALUE + CDS_SOLD_NEG_FAIR_VALUE

DERI_PAY = ["RCONC012", "RCFDC012"]    # RC-G 4.d
DERI_INT_RATE_FAIR_VALUE_NEG_HOLD_FOR_TRADING = ['RCON8737', "RCFD8737"] # RC-L 15.a.(1)
DERI_FOR_EXCH_FAIR_VALUE_NEG_HOLD_FOR_TRADING = ['RCON8738', "RCFD8738"] # RC-L 15.a.(1)
DERI_EQU_DERI_FAIR_VALUE_NEG_HOLD_FOR_TRADING = ['RCON8739', "RCFD8739"] # RC-L 15.a.(1)
DERI_COM_OTHE_FAIR_VALUE_NEG_HOLD_FOR_TRADING = ['RCON8740', "RCFD8740"] # RC-L 15.a.(1)

DERI_INT_RATE_FAIR_VALUE_NEG_HOLD_NOT_TRADING = ['RCON8745', "RCFD8745"] # RC-L 15.b.(1)
DERI_FOR_EXCH_FAIR_VALUE_NEG_HOLD_NOT_TRADING = ['RCON8746', "RCFD8746"] # RC-L 15.b.(1)
DERI_EQU_DERI_FAIR_VALUE_NEG_HOLD_NOT_TRADING = ['RCON8747', "RCFD8747"] # RC-L 15.b.(1)
DERI_COM_OTHE_FAIR_VALUE_NEG_HOLD_NOT_TRADING = ['RCON8748', "RCFD8748"] # RC-L 15.b.(1)


# Part 10 Unused commitment
# g lcr_uc_outflow=  0.05*uc_he+  0.05*uc_ccard+  0.10*uc_cre+  1.00*uc_sec_uw+   1.00*uc_other+  * letter of credit *  0.05*lc
	
UC_HOME_EQUITY = ["RCFD3814", "RCON3814"]   # RC-L, 1.a.
UC_CRED_CARD = ["RCFD3815", "RCON3815"]     # RC-L, 1.b.

UC_CRE_FAMILY_RESIDENTIAL = ["RCFDF164", "RCONF164"]
UC_CRE_COMMERCIAL_ESTATE = ["RCFDF165", "RCONF165"]
UC_CRE_NOT_SECURED = ["RCFD6550", "RCON6550"]
# UC_CRE = UC_CRE_FAMILY_RESIDENTIAL + UC_CRE_COMMERCIAL_ESTATE + UC_CRE_NOT_SECURED
# TODO verify this
UC_SEC_UW = ["RCFD3817", "RCON3817"]
# uc_other=max(rcfd3818, rcon3818)+max(rcfdj457, rconj457)+max(rcfdj458, rconj458)+max(rcfdj459, rconj459)
UC_COMMERCIAL = ["RCFDJ457", "RCONJ457"]
UC_FINANCIAL = ["RCFDJ458", "RCONJ458", "RCFDVP10", "RCONPV10"]
UC_OTHER = ["RCFDJ459", "RCONJ459"]

FINANCIAL_LETTER_OF_CREDIT = ["RCFD3819", "RCON3819"]
PERFORMANCE_LETTER_OF_CREDIT = ["RCFD3821", "RCON3821"]
COMMERCIAL_LETTER_OF_CREDIT = ["RCFD3411", "RCON3411"]
