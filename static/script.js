document.addEventListener(
    "DOMContentLoaded",
    function () {


        // ====================================================
        // FILE NAME
        // ====================================================

        const fileInput =
            document.getElementById(
                "document"
            );


        const fileName =
            document.getElementById(
                "file-name"
            );


        if (fileInput) {

            fileInput.addEventListener(
                "change",
                function () {


                    if (
                        fileInput.files.length > 0
                    ) {

                        fileName.textContent =
                            fileInput.files[0].name;

                    } else {

                        fileName.textContent =
                            "No file selected";

                    }

                }
            );

        }



        // ====================================================
        // FORM SUBMIT
        // ====================================================

        const form =
            document.getElementById(
                "upload-form"
            );


        const loading =
            document.getElementById(
                "loading"
            );


        const button =
            document.getElementById(
                "check-button"
            );


        if (form) {

            form.addEventListener(
                "submit",
                function () {


                    if (loading) {

                        loading.style.display =
                            "block";

                    }


                    if (button) {

                        button.disabled =
                            true;

                        button.textContent =
                            "Analyzing...";

                    }

                }
            );

        }

    }
);