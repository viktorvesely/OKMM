var table = document.getElementById("table_root");

const question_id = 1;
const answer_id = 2;
const label_id = 3;
const mark_id = 4;
const n_id = 5;


function labelByLabel(cells, maxPoints) {
    let label = cells[label_id].innerText;
    let value = null;
    if (label === "nesprávne") {
        value = 0;
    } else if (Number.isInteger(Number.parseInt(label[0]))) {
        let processed = label.replace(",", ".");
        value = Number.parseFloat(processed);
    } else if (label === "správne") {
        value = maxPoints;
    } else if (label === "") {
        value = 0;  
    } else {
        debugger;
        throw new Error(`Non-valid label: ${label}`);
    }
    return value;
}


function labelByMark(cells, maxPoints) {
    let mark = cells[mark_id].innerText;
    let value = null;
    if (mark === "nie") {
        value = 0;
    } else if (mark === "áno") {
        value = maxPoints;
    } else if (mark === "") {
        value = 0;
    } else {
        debugger;
        throw new Error(`Non-valid mark: ${mark}`);
    }
    return value;
}


function parseTable(maxPoints=1, labelResolver=null) {
    let data = {answer: [], label: [], n: [], id: []};

    let rows = table.querySelectorAll(`tr.st_riadok`);

    rows.forEach(row => {
        let cells = row.querySelectorAll("td");

        if (cells.length < 3) return;

        let value = labelResolver(cells, maxPoints);

        data.answer.push(cells[answer_id].innerText);
        data.label.push(value);
        data.n.push(Number.parseInt(cells[n_id].innerText));
        data.id.push(Number.parseInt(cells[question_id].innerText));
    });

    return data;
} 

function downloadObjectAsJson(exportObj, exportName) {
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj));
    var downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href",     dataStr);
    downloadAnchorNode.setAttribute("download", exportName);
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

let data = parseTable(3, labelByLabel); 
downloadObjectAsJson(data, `q_2_k_${data.id[0]}.json`);