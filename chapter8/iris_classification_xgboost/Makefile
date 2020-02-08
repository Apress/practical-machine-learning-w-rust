java_run: lib
	javac IrisClassificationXgboost.java && java -Djava.library.path=iris_classification_library/target/debug/ IrisClassificationXgboost

.PHONY: lib

javah:
	javah IrisClassificationXgboost

lib:
	cd iris_classification_library && cargo build
