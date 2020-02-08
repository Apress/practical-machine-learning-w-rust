use rusted_cypher;
use rusted_cypher::{GraphClient, Statement, GraphError};
use std::iter::repeat;

fn main() -> Result<(), Box<GraphError>> {
    // let graph = GraphClient::connect(
    //     "http://neo4j:neo4j@localhost:7474/db/data");
    let graph = GraphClient::connect(
        "http://localhost:7474/db/data")?;

    let mut query = graph.query();

    // create index
    let statement1 = Statement::new(
        "CREATE CONSTRAINT ON (m:Movie) ASSERT m.id IS UNIQUE;");
    let statement2 = Statement::new(
        " CREATE CONSTRAINT ON (u:User) ASSERT u.id IS UNIQUE;"
    );
    let statement3 = Statement::new(
        " CREATE CONSTRAINT ON (g:Genre) ASSERT g.name IS UNIQUE;"
    );

    query.add_statement(statement1);
    query.add_statement(statement2);
    query.add_statement(statement3);

    query.send()?;

    // import movies.csv
    graph.exec(
        "USING PERIODIC COMMIT LOAD CSV WITH HEADERS \
        FROM \"http://10.0.1.43:8000/movies.csv\" AS line \
        WITH line, SPLIT(line.genres, \"|\") AS Genres \
        CREATE (m:Movie { id: TOINTEGER(line.`movieId`), title: line.`title` }) \
        WITH Genres \
        UNWIND RANGE(0, SIZE(Genres)-1) as i \
        MERGE (g:Genre {name: UPPER(Genres[i])}) \
        CREATE (m)-[r:GENRE {position:i+1}]->(g);"
    )?;

    // import ratings.csv
    graph.exec(
        " USING PERIODIC COMMIT LOAD CSV WITH HEADERS \
        FROM \"http://10.0.1.43:8000/ratings.csv\" AS line \
        WITH line \
        MATCH (m:Movie { id: TOINTEGER(line.`movieId`) }) \
        MATCH (u:User { id: TOINTEGER(line.`userId`) }) \
        CREATE (u)-[r:RATING {rating: TOFLOAT(line.`rating`)}]->(m);"
    )?;

    // import tags
    graph.exec(
        " USING PERIODIC COMMIT LOAD CSV WITH HEADERS \
        FROM \"http://10.0.1.43:8000/tags.csv\" AS line \
        WITH line \
        MATCH (m:Movie { id: TOINTEGER(line.`movieId`) }) \
        MERGE (u:User { id: TOINTEGER(line.`userId`) }) \
        CREATE (u)-[r:TAG {tag: line.`tag`}]->(m);"
    )?;

    let result = graph.exec(
        "MATCH (u:User {id: 119}) RETURN u.id")?;

    assert_eq!(result.data.len(), 1);

    for row in result.rows() {
        let id: u16 = row.get("u.id")?;
        println!("user id: {}", id);
    }

    // understand the shortest paths between all

    let all_users = graph.exec(
        "MATCH (u:User) RETURN u.id")?;
    let all_users: Vec<u32> = all_users.rows().map(|x| x.get("u.id").unwrap()).collect();

    let mut length_of_paths = vec![];
    for (u1, u2) in all_users.iter()
            .enumerate()
            .flat_map(|(i, val)| repeat(val).zip(all_users.iter().skip(i + 1))) {
        let statement = format!(
            "MATCH (n:User) where n.id IN [{user1}, {user2}]
            WITH collect(n) as nodes
            UNWIND nodes as n
            UNWIND nodes as m
            WITH * WHERE id(n) < id(m)
            MATCH path = allShortestPaths( (n)-[*..4]-(m) )
            RETURN min(length(path))", user1=u1, user2=u2);
        let min_paths = graph.exec(statement)?;
        let min_paths: Vec<Option<u32>> = min_paths.rows().map(|x| x.get("min(length(path))").unwrap()).collect();
        match min_paths[0] {
            Some(mp) => {length_of_paths.push((u1, u2, mp)); ()},
            _ => (),
        };
    }
    println!("{:?}", length_of_paths);

    Ok(())
}