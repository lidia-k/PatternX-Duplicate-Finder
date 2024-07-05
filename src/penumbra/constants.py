SP_COLS = {
    'Request Type': 'type',
    'HCP Full Name': 'fullname',
    'NPI Number': 'npi',
    'HCP Category': 'category',
    'HCP Specialty': 'specialty',
    'HCP Institution / Customer Name': 'org',
    'HCP Institution': 'org',
    'SAP Customer ID': 'sap_no',
    'Institution City': 'city',
    'Institution State': 'state',
    'HCP Country': 'country',
    'HCP Email ': 'email',
    'HCC ID': 'hid',
    'HCP NPI#': 'npi',
    'SAP Supplier ID': 'sap_no',
    'Presentation Title': 'title',
    'Country': 'country2',
}

PO_COLS = {
    'First Name': 'fname',
    'Last Name': 'lname',
    'Full Name': 'fullname',
    'National Physician ID': 'npi',
    'National Physician ID/RPPS ID': 'npi',
    'Email Address': 'email',
    'Quickbase Record ID#':  'qb_id',
    'HCP Category': 'category',
    'Payments Made To:': 'payments_to',
    'SAP Number': 'sap_no',
    'SAP Entity Name': 'sap_name',
    'State/Region/Province': 'state1',
    'State/Region': 'state2',
    'State License #': 'license',
    'License State (US)': 'lic_state',
    'Focus Area': 'fc_area',
    'Taxonomy Code': 'tax_code',
    'Payment Currency': 'currency',
    'Primary Address': 'addr1',
    'Mailing Address': 'addr2',
    'Primary Organization': 'org',
    'Primary Organization Type': 'org_type',
}

PO_VC_COLS = {
    'Full Name': 'fullname',
    'first_name': 'fname',
    'middle_name': 'mname',
    'last_name': 'lname',
    'country_code': 'country',
    'address_1': 'addr1',
    'address_2': 'addr2',
    'lisc': 'license'
}

INT_COLS = ['npi', 'qb_id', 'sap_no', 'license']

EDGE_TYPES = [
    'npi',
    'fullname_email',
    'fullname_sap_no',
    'fullname_qb_id'
]

COLS_TO_USE = ['uid', 'fname', 'lname', 'fullname', 'npi', 
               'country', 'speciality', 'email', 'sap_no', 'qb_id']
