LOAD CSV WITH HEADERS FROM '{file_path}' AS row
CREATE (s:Speaker {{
    fullname: row.fullname,
    npi: toInteger(row.npi),
    category: row.category,
    specialty: row.specialty,
    org: row.org,
    sapNo: toInteger(row.sap_no),
    city: row.city,
    state: row.state,
    country: row.country,
    email: row.email,
    franchise: row.franchise,
    id: row.id
}});
