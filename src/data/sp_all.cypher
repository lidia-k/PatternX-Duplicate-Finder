LOAD CSV WITH HEADERS FROM '{file_path}' AS row
CREATE (s:Speaker {{
    franchise: row.franchise,
    fullname: row.fullname,
    fname: row.fname,
    lname: row.lname,
    npi: toInteger(row.npi),
    sap_no: toInteger(row.sap_no),
    category: row.category,
    specialty: row.specialty,
    org: row.org,
    country: row.country,
    title: row.title,
    country2: row.country2,
    uid: row.uid,
    text: row.text
}});
