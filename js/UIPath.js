const robot = UIPathRobot.init();

getProcesses(): Promise<Array<RobotProcess>>;

robot.getProcesses()
.then(result => {
  for (let i = 0; i < result.length; i++){
    console.log(result[i].name);
  }
} err => {
  console.log(err);
});

// start job

let arguments = {
  "inputs1": 23,
  "inputs2": 23,
  "operation": "add"
};

robot.getProcesses()
.then(processes => {
  let calculatorProcess = processes.find(p => p.name.includes("Calculator"))
  let job = new Job(calculatorProcess.id, arguments);
  robot.startJob(job).then(result => {
    console.log(result.Sum);
  }, err => {
    console.log(err);
  })
}, err => {
  console.log(err)
}};

on(eventName: string, callback: (argument?: any) => void): void;

robot.on('consent-prompt', (consentCode) => {console.log(consentCode)});
robot.on('missing-components', [] => {console.log('Missing components')});

class RobotProcess {
  start: (inArguments?: any) => JobPromise;
}
