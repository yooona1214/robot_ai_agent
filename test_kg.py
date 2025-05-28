from neo4j import GraphDatabase

# Neo4j 드라이버 설정
uri = "bolt://localhost:7687"
user = "neo4j"
password = "test"
driver = GraphDatabase.driver(uri, auth=(user, password))

def run_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]

# Neo4j 쿼리 예시
query = "MATCH (n) RETURN n LIMIT 5"
result = run_query(query)
print(result)
