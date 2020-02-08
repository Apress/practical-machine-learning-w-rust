class IrisClassificationXgboost {
    private static native void fit();
    private static native String predict();

    static {
        // This actually loads the shared object that we'll be creating.
        // The actual location of the .so or .dll may differ based on your
        // platform.
        System.loadLibrary("iris_classification_xgboost");
    }

    // The rest is just regular ol' Java!
    public static void main(String[] args) {
        IrisClassificationXgboost.fit();
        String predictions = IrisClassificationXgboost.predict();
        System.out.println(predictions);
    }
}