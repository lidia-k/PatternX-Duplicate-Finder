import requests, pandas, json, os, argparse
from neo4j import GraphDatabase
parser  = argparse.ArgumentParser(description='Run different functions based on input parameters.')
parser.add_argument('--lookup', dest="lookup", action="store_true")
parser.add_argument('--insert_npi', action="store_true")
args    = parser.parse_args()
dadi    = ""
upass   = ("neo4j", "password")
# outfile = dadi + "sp-retrievedNpi.csv"
# infile  = dadi + "sp-noNpi.json"
outfile = dadi + "po-retrievedNpi.csv"
infile  = dadi + "po-noNpi.json"
url     = "https://npiregistry.cms.hhs.gov/api/"

#-------------------------------------------------------------------------------
# intent: perform npi registry lookup given a list of names
# input : results from this neo4j query, downloaded to json file
#         match (n) where n.npi is null and ( n.fname is not null and n.lname is not null ) 
#         return n.uid, n.fname, n.lname
#         speaker file does not separate fname & lname, so use:  .. n.fullname is not null ..
# output: a csv file like this:  
#         {uid, first_name, last_name, result}
#-------------------------------------------------------------------------------
def lookupNPI():
    if os.path.exists( outfile ):  os.remove( outfile )
    results = []
    with open( infile, "r", encoding='utf-8-sig') as file:  data = json.load(file)
    for i in range( len(data) ):
        pram = { 'first_name': data[i]['n.fname'], 'last_name': data[i]['n.lname'], 'version':2.1 }
        try: 
           response = requests.get( url, params=pram )
           response.raise_for_status()  
           xx = response.json().get("results", [])
        except Exception as e:   print(f"Error checking NPI {npi}: {e}")
        results.append({'uid': data[i]['n.uid'], 'first_name': data[i]['n.fname'], 'last_name': data[i]['n.lname'], 'result': json.dumps(xx)})
        print(   data[i]['n.uid'], data[i]['n.fname'], data[i]['n.lname'] , " -- ", xx )
    df = pandas.DataFrame(results)
    df.to_csv(outfile, index=False)
#-------------------------------------------------------------------------------
# Intent: Only insert NPI into Neo4J of qualified rows:
# - There is only 1 row including fname, lname. means ignore rows where fname and lname overlap.
# - this row only returns 1 result from Npi Registry Lookup. means if more than 1 npi is received from Npi Registry Lookup then ignore.
# Input: csv file outfile from lookupNPI() function
# Output: csv file (outfile + "_result.csv" ) contains updated rows with corresponding npi
#-------------------------------------------------------------------------------
def insertNpi():
    try:   driver = GraphDatabase.driver( "bolt://localhost:7687", auth=upass )
    except Exception as e:  print('error.  Is the neo4j database docker container running?')
    baseQuery = "match (n) where n.uid = \'{}\' set n.npi = \'{}\'"
    df = pandas.read_csv(outfile)
    print(df.columns)
    df["result"] = df["result"].apply(lambda x: json.loads(x))
    df = df[df['result'].map(len) == 1] # ignore if more than 1 npi is received from Npi Registry Lookup 
    grouped = df.groupby(["first_name", "last_name"])
    df["npi"] = pandas.NA; df = df.set_index("uid", drop=False)
    updated_npi = 0
    with driver.session() as session:
        for name, group in grouped:
            if len(group) == 1: # ignore rows where fname and lname overlap.
                for row_index, row in group.iterrows():
                    npi = row["result"][0]["number"]
                    df.loc[row["uid"], "npi"] = npi
                    updated_npi += 1
                    update = baseQuery.format(row["uid"], npi)
                    # # update = f"""MATCH (n) where n.uid = "{row["uid"]}" REMOVE n.npi RETURN n"""
                    session.run(update)
                    # npis.append(npi)
                    print("Updated node.uid = {} with npi = {}".format(row["uid"], npi))
    print(f"Total - Updated node: {updated_npi}")
    df = df[df["npi"].notna()]
    df.to_csv(outfile+"_result.csv", index=False)

if args.lookup: lookupNPI()
if args.insert_npi: insertNpi()
